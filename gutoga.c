#include <float.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gutoga.h"

struct gutoga_mutex_t *gutoga_mutex_new()
{
	struct gutoga_mutex_t *lock;
	lock = calloc(1, sizeof(*lock));
	if (lock == NULL) {
		return NULL;
	}
	omp_init_lock(&lock->l);
	return lock;
}

void gutoga_mutex_destroy(struct gutoga_mutex_t *lock)
{
	omp_destroy_lock(&lock->l);
	free(lock);
}

void gutoga_mutex_lock(struct gutoga_mutex_t *lock)
{
	omp_set_lock(&lock->l);
}

void gutoga_mutex_unlock(struct gutoga_mutex_t *lock)
{
	omp_unset_lock(&lock->l);
}

int gutoga_cache_add(gutoga_cache_t *cache, BYTE *hash, void *ptr, double score, bool *exists)
{
	*exists = false;
	if (cache == NULL) {
		return GUTOGA_SUCCESS;
	}
	gutoga_mutex_lock(cache->lock);
	struct gutoga_cache_entry_t *v = NULL;
	HASH_FIND(hh, cache->records, hash, SHA256_BLOCK_SIZE, v);
	*exists = v != NULL;
	if (v == NULL) {
		v = calloc(1, sizeof(*v));
		if (v == NULL) {
			gutoga_mutex_unlock(cache->lock);
			return GUTOGA_ERR_BAD_ALLOC_CACHE;
		}
		v->ptr = ptr;
		v->refs = 1;
		v->score = score;
		memcpy(v->hash, (void *)hash, SHA256_BLOCK_SIZE);
		HASH_ADD(hh, cache->records, hash, SHA256_BLOCK_SIZE, v);
	}
	v->ttl = cache->ttl;
	gutoga_mutex_unlock(cache->lock);
	return GUTOGA_SUCCESS;
}

void gutoga_cache_get(gutoga_cache_t *cache, BYTE *hash, void **ptr, double *score)
{
	if (cache == NULL) {
		return;
	}
	gutoga_mutex_lock(cache->lock);
	struct gutoga_cache_entry_t *v = NULL;
	HASH_FIND(hh, cache->records, hash, SHA256_BLOCK_SIZE, v);
	if (v == NULL) {
		*ptr = NULL;
	}
	else {
		*ptr = v->ptr;
		if (score != NULL) {
			*score = v->score;
		}
	}
	gutoga_mutex_unlock(cache->lock);
}

void gutoga_cache_ref(gutoga_cache_t *cache, BYTE *hash)
{
	if (cache == NULL) {
		return;
	}
	gutoga_mutex_lock(cache->lock);
	struct gutoga_cache_entry_t *v = NULL;
	HASH_FIND(hh, cache->records, hash, SHA256_BLOCK_SIZE, v);
	if (v != NULL) {
		v->refs++;
	}
	gutoga_mutex_unlock(cache->lock);
}

void gutoga_cache_del(gutoga_cache_t *cache, BYTE *hash)
{
	if (cache == NULL) {
		return;
	}
	gutoga_mutex_lock(cache->lock);

	struct gutoga_cache_entry_t *v = NULL;
	HASH_FIND(hh, cache->records, hash, SHA256_BLOCK_SIZE, v);
	if (v != NULL) {
		HASH_DEL(cache->records, v);
		free(v);
	}

	gutoga_mutex_unlock(cache->lock);
}


void gutoga_cache_epoch(void *context, gutoga_cache_t *cache, gutoga_cleanup_fn cleanup)
{
	if (cache == NULL) {
		return;
	}
	gutoga_mutex_lock(cache->lock);
	struct gutoga_cache_entry_t *v = NULL;
	struct gutoga_cache_entry_t *tmp = NULL;
	HASH_ITER(hh, cache->records, v, tmp)
	{
		if (v->refs == 0) {
			v->ttl--;
		}
		v->refs = 0;
		if (v->ttl == 0) {
			cleanup(context, v->ptr);
			HASH_DEL(cache->records, v);
			free(v);
		}
	}
	gutoga_mutex_unlock(cache->lock);
}

gutoga_cache_t *gutoga_cache_new(const int ttl)
{
	gutoga_cache_t *cache;
	cache = calloc(1, sizeof(*cache));
	if (cache == NULL) {
		return NULL;
	}
	cache->lock = gutoga_mutex_new();
	if (cache->lock == NULL) {
		free(cache);
		cache = NULL;
	}
	cache->ttl = ttl;
	return cache;
}

void gutoga_cache_destroy(void *context, gutoga_cache_t *cache, gutoga_cleanup_fn cleanup)
{
	if (cache == NULL) {
		return;
	}
	gutoga_mutex_destroy(cache->lock);
	struct gutoga_cache_entry_t *v = NULL;
	struct gutoga_cache_entry_t *tmp = NULL;
	HASH_ITER(hh, cache->records, v, tmp)
	{
		if (cleanup != NULL) {
			cleanup(context, v->ptr);
		}
		HASH_DEL(cache->records, v);
		free(v);
	}
	cache->records = NULL;
	free(cache);
}

void do_print(gutoga_node_t *node, char *padding, char *buf, int has_right_sibling)
{
	printf("\n");
	printf("%s", padding);
	printf("%s", buf);
	if (has_right_sibling) {
		printf("%d:%d:%d(l", node->id, node->left_edges, node->right_edges);
		printf(")");
	}
	else {
		printf("%d:%d:%d(r", node->id, node->left_edges, node->right_edges);
		printf(")");
	}
	char padding1[1024] = {0};
	if (has_right_sibling == 1) {
		sprintf(padding1, "%s│  ", padding);
	}
	else {
		sprintf(padding1, "%s   ", padding);
	}

	char *buf_left = "├──";
	char *buf_right = "└──";
	if (node->right == NULL) {
		buf_left = "└──";
	}

	if (node->left != NULL) {
		do_print(node->left, padding1, buf_left, node->right != NULL);
	}
	if (node->right != NULL) {
		do_print(node->right, padding1, buf_right, 0);
	}
}

void printer(gutoga_tree_t t)
{
	if (t.root == NULL) {
		return;
	}
	gutoga_node_t *root = t.root;
	printf("%d:(%d:%d)", root->id, root->left_edges, root->right_edges);
	char *buf_left = "├──";
	char *buf_right = "└──";
	if (root->right == NULL) {
		buf_left = "└──";
	}
	if (root->left != NULL) {
		do_print(root->left, "", buf_left, root->right != NULL);
	}
	if (root->right != NULL) {
		do_print(root->right, "", buf_right, 0);
	}
	printf("\n");
}

int gutoga_tree_split_edge(gutoga_tree_t *t, gutoga_node_t *from, gutoga_node_t *to, const int id)
{
	gutoga_node_t *new_node = &t->nodes[t->nodes_len++];
	new_node->id = -1;
	gutoga_node_t *new_leaf = &t->nodes[t->nodes_len++];
	new_leaf->id = id;

	if (from->left == to) {
		from->left = new_node;
		from->left_edges += 2;
	}
	else {
		from->right = new_node;
		from->right_edges += 2;
	}

	new_node->left = to;
	new_node->right = new_leaf;
	new_node->right_edges = 1;
	new_node->left_edges = to->left_edges + to->right_edges + 1;

	return GUTOGA_SUCCESS;
}

int gutoga_tree_split(gutoga_tree_t *t, gutoga_node_t *node, const int edge, const int id)
{
	if (edge < node->left_edges) {
		if (edge == node->left_edges - 1) {
			return gutoga_tree_split_edge(t, node, node->left, id);
		}
		else {
			node->left_edges += 2;
			return gutoga_tree_split(t, node->left, edge, id);
		}
	}
	else {
		if (edge == node->left_edges + node->right_edges - 1) {
			return gutoga_tree_split_edge(t, node, node->right, id);
		}
		else {
			node->right_edges += 2;
			return gutoga_tree_split(t, node->right, edge - node->left_edges, id);
		}
	}
}

int gutoga_tree_add_node(gutoga_tree_t *t, const int edge, const int id)
{
	if (edge < t->root->left_edges + t->root->right_edges) {
		return gutoga_tree_split(t, t->root, edge, id);
	}

	gutoga_node_t *new_root = &t->nodes[t->nodes_len++];
	gutoga_node_t *new_node = &t->nodes[t->nodes_len++];

	new_root->id = -1;
	new_root->left = t->root;
	new_root->right = new_node;

	new_root->left_edges = t->root->left_edges + t->root->right_edges + 1;
	new_root->right_edges = 1;
	new_node->id = id;
	t->root = new_root;

	return GUTOGA_SUCCESS;
}

static int gutoga_tree_num_nodes(const int size)
{
	return 2 * size - 1;
}

int gutoga_tree_init_empty(gutoga_tree_t *t, const int size)
{
	t->root = NULL;
	t->has_score = false;
	t->score = 0;
	t->size = size;
	int nodes = gutoga_tree_num_nodes(size);
	if (nodes <= 3) {
		return GUTOGA_ERR_BAD_TREE_SIZE;
	}
	t->enc = calloc(size - 2, sizeof(*t->enc));
	if (t->enc == NULL) {
		return GUTOGA_ERR_BAD_ALLOC_TREE;
	}

	t->nodes = calloc(nodes, sizeof(*t->nodes));
	if (t->nodes == NULL) {
		return GUTOGA_ERR_BAD_ALLOC_TREE;
	}
	return GUTOGA_SUCCESS;
}

void gutoga_tree_default(gutoga_tree_t *t)
{
	/*   -1    */
	/*   / \   */
	/*  /   \  */
	/* 0     1 */

	gutoga_node_t root = {
	    .id = -1,
	    .left = t->nodes + 1,
	    .right = t->nodes + 2,
	    .left_edges = 1,
	    .right_edges = 1,
	};
	t->root = t->nodes;
	t->nodes[0] = root;
	t->nodes[1].id = 0;
	t->nodes[2].id = 1;
	t->nodes_len = 3;
	t->has_score = false;
}

int gutoga_tree_init(gutoga_tree_t *t, const int size)
{
	int rc = gutoga_tree_init_empty(t, size);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	gutoga_tree_default(t);

	return GUTOGA_SUCCESS;
}

static int gutoga_tree_sequence(gutoga_tree_t *t)
{
	for (int i = 0; i < t->size - 2; i++) {
		int rc = gutoga_tree_add_node(t, t->enc[i], i + 2);
		if (rc != 0) {
			return rc;
		}
	}
	return GUTOGA_SUCCESS;
}

int gutoga_tree_random(void *context, gutoga_tree_t *t, const int size, gutoga_random_int_fn rnd)
{
	int rc = gutoga_tree_init(t, size);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}

	for (int i = 0; i < size - 2; i++) {
		t->enc[i] = rnd(context, 2 * i + 3);
	}
	return gutoga_tree_sequence(t);
}

int gutoga_tree_init_sequence(void *context, gutoga_tree_t *t, const int *seq, const int size)
{
	int rc = gutoga_tree_init(t, size);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	memmove(t->enc, seq, (size - 2) * sizeof(*seq));
	return gutoga_tree_sequence(t);
}

int gutoga_tree_clone(void *context, gutoga_tree_t *dst, gutoga_tree_t src)
{
	int rc = gutoga_tree_init_sequence(context, dst, src.enc, src.size);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	dst->has_score = src.has_score;
	dst->score = src.score;
	memcpy(dst->root->hash, src.root->hash, SHA256_BLOCK_SIZE);
	return GUTOGA_SUCCESS;
}

void gutoga_tree_destroy(void *context, gutoga_ga_t *ga, gutoga_tree_t t)
{
	if (t.nodes != NULL) {
		free(t.nodes);
	}
	if (t.enc != NULL) {
		free(t.enc);
	}
}
int gutoga_node_align(void *context, gutoga_ga_t *ga, gutoga_node_t *node, void **data, double *score)
{
	if (node->id != -1) {
		return ga->cfg->id_to_ptr(context, node->id, data);
	}

	if ((node->left == NULL) ^ (node->right == NULL)) {
		// By definition all internal nodes have exactly two children.
		return GUTOGA_ERR_BAD_TREE;
	}

	gutoga_cache_get(ga->cache, node->hash, data, score);
	if (*data != NULL) {
		return GUTOGA_SUCCESS;
	}

	void *left = NULL;
	void *right = NULL;
	int rc = gutoga_node_align(context, ga, node->left, &left, NULL);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	rc = gutoga_node_align(context, ga, node->right, &right, NULL);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}

	rc = ga->cfg->align(context, left, right, data, score);
	double sc = 0;
	if (score != NULL) {
		sc = *score;
	}
	bool exists = false;
	gutoga_cache_add(ga->cache, node->hash, *data, sc, &exists);
	if (exists) {
		ga->cfg->cleanup(context, *data);
		gutoga_cache_get(ga->cache, node->hash, data, score);
	}

	return rc;
}

void gutoga_node_hash(gutoga_ga_t *ga, gutoga_node_t *node)
{
	SHA256_CTX ctx;
	sha256_init(&ctx);
	if (node->id != -1) {
		sha256_update(&ctx, (BYTE *)&(node->id), sizeof(node->id));
		sha256_final(&ctx, node->hash);
	}
	else {
		gutoga_node_hash(ga, node->left);
		gutoga_node_hash(ga, node->right);

		sha256_update(&ctx, node->left->hash, SHA256_BLOCK_SIZE);
		sha256_update(&ctx, node->right->hash, SHA256_BLOCK_SIZE);

		sha256_final(&ctx, node->hash);
	}
	/* printf("hash: %02X %02X %02X\n", node->hash[0], node->hash[1], node->hash[2]); */
	gutoga_cache_ref(ga->cache, node->hash);
}

struct tree_order_retriever_t {
	int *order;
};

void gutoga_tree_traverse(const gutoga_node_t *node, struct tree_order_retriever_t *dst)
{
	if (node->left != NULL) {
		gutoga_tree_traverse(node->left, dst);
	}
	if (node->id != -1) {
		*dst->order = node->id;
		dst->order++;
	}
	if (node->right != NULL) {
		gutoga_tree_traverse(node->right, dst);
	}
}

int *gutoga_tree_order(const gutoga_tree_t t)
{
	int *dst = calloc(t.size, sizeof(int));
	if (dst == NULL) {
		return NULL;
	}

	struct tree_order_retriever_t r = {
	    .order = dst,
	};

	gutoga_tree_traverse(t.root, &r);
	return dst;
}


gutoga_population_t *gutoga_population_new(int size)
{
	gutoga_population_t *p = calloc(1, sizeof(struct gutoga_population_t));
	if (p == NULL) {
		return NULL;
	}
	p->size = 0;
	p->cap = size * 2;
	p->trees = calloc(p->cap, sizeof(*p->trees));
	if (p->trees == NULL) {
		free(p);
		return NULL;
	}
	return p;
}

void gutoga_population_destroy(void *context, gutoga_ga_t *ga, gutoga_population_t *p)
{
	for (int i = 0; i < p->size; i++) {
		gutoga_tree_destroy(context, ga, p->trees[i]);
	}
	if (p->trees != NULL) {
		free(p->trees);
		p->trees = NULL;
	}
	free(p);
}

void gutoga_population_reset(void *context, gutoga_ga_t *ga, gutoga_population_t *p)
{
	for (int i = 0; i < p->size; i++) {
		gutoga_tree_destroy(context, ga, p->trees[i]);
	}
	p->size = 0;
}

int gutoga_population_append(struct gutoga_population_t *p, const gutoga_tree_t t)
{
	if (p->size >= p->cap) {
		p->cap *= 2;
		gutoga_tree_t *m = realloc(p->trees, p->cap * sizeof(*p->trees));
		if (m == NULL) {
			return GUTOGA_ERR_BAD_ALLOC_POPULATION;
		}
		p->trees = m;
	}

	p->trees[p->size] = t;
	p->size++;
	return GUTOGA_SUCCESS;
}

int gutoga_tree_score(void *context, gutoga_ga_t *ga, gutoga_tree_t *tree, const bool force, double *fitness)
{
	int rc = GUTOGA_SUCCESS;
	if (!tree->has_score) {
		void *dst;
		rc = gutoga_node_align(context, ga, tree->root, &dst, &tree->score);
		if (rc == GUTOGA_ERR_FITNESS_SKIPPABLE) {
			tree->score = 0;
		}
	}
	tree->has_score = true;
	if (fitness != NULL) {
		*fitness = tree->score;
	}
	return rc;
}

int gutoga_tree_fitness(void *context, gutoga_ga_t *ga, gutoga_tree_t *tree, double *fitness)
{
	return gutoga_tree_score(context, ga, tree, false, fitness);
}

int gutoga_tree_force_fitness(void *context, gutoga_ga_t *ga, gutoga_tree_t *tree, void **dst)
{
	int rc = gutoga_node_align(context, ga, tree->root, dst, NULL);
	gutoga_cache_del(ga->cache, tree->root->hash);
	return rc;
}

static int points_bin_search(const double *points, const int size, const double point)
{
	int l = 0;
	int r = size;
	while (l < r) {
		int h = (l + r) >> 1;
		if (points[h] < point) {
			l = h + 1;
		}
		else {
			r = h;
		}
	}
	return l;
}

int gutoga_selection_roulette_wheel(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats)
{
	double start = omp_get_wtime();
	gutoga_population_reset(context, ga, ga->mates);

	double *points = calloc(ga->pop->size, sizeof(double));
	if (points == NULL) {
		return GUTOGA_ERR_BAD_ALLOC_SELECTION;
	}
	int rc;
	double sum = 0.0;
	for (int i = 0; i < ga->pop->size; i++) {
		double fitness = 0.0;
		rc = gutoga_tree_fitness(context, ga, ga->pop->trees + i, &fitness);
		if (rc != GUTOGA_SUCCESS && rc != GUTOGA_ERR_FITNESS_SKIPPABLE) {
			goto RW_SELECTION_ERROR;
		}
		sum += fitness;
		points[i] = sum;
	}

	for (int i = 0; i < ga->pop->size; i++) {
		double rnd = ga->cfg->rand_double(context) * sum;
		int ind = points_bin_search(points, ga->pop->size, rnd);
		gutoga_tree_t tree;
		rc = gutoga_tree_clone(context, &tree, ga->pop->trees[ind]);
		if (rc != GUTOGA_SUCCESS) {
			goto RW_SELECTION_ERROR;
		}
		rc = gutoga_population_append(ga->mates, tree);
		if (rc != GUTOGA_SUCCESS) {
			goto RW_SELECTION_ERROR;
		}
	}

	rc = GUTOGA_SUCCESS;
RW_SELECTION_ERROR:
	free(points);
	ga->cfg->logger(context, "selection elapsed: %f; (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf)\n", omp_get_wtime() - start, stats->min, stats->max, stats->avg, stats->best.score);
	return rc;
}

void gutoga_population_shuffle(void *context, gutoga_population_t *pop, gutoga_random_int_fn rnd)
{
	for (int i = pop->size - 1; i > 0; i--) {
		size_t j = rnd(context, i + 1);
		gutoga_tree_t t = pop->trees[j];
		pop->trees[j] = pop->trees[i];
		pop->trees[i] = t;
	}
}

int gutoga_crossover_one_point(void *context,
			       const gutoga_tree_t parent_a, const gutoga_tree_t parent_b,
			       gutoga_tree_t *child_a, gutoga_tree_t *child_b,
			       gutoga_random_int_fn rnd)
{

	int rc = gutoga_tree_init(child_a, parent_a.size);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	rc = gutoga_tree_init(child_b, parent_a.size);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}

	int point = rnd(context, parent_a.size - 2);
	for (int i = 0; i < point; i++) {
		child_a->enc[i] = parent_a.enc[i];
		child_b->enc[i] = parent_b.enc[i];
	}
	for (int i = point; i < parent_a.size - 2; i++) {
		child_a->enc[i] = parent_b.enc[i];
		child_b->enc[i] = parent_a.enc[i];
	}

	rc = gutoga_tree_sequence(child_a);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	return gutoga_tree_sequence(child_b);
}

int gutoga_mutation_adaptive_prob(void *context, const gutoga_tree_t t, gutoga_stats_t *stats, double *prob)
{
	double cur = t.score;
	if (cur < stats->avg) {
		*prob = 0.5;
	}
	else {
		*prob = 0.5 * ((stats->max - cur) / (stats->max - stats->avg));
	}
	return GUTOGA_SUCCESS;
}

int gutoga_crossover_adaptive_prob(void *context, const gutoga_tree_t a, const gutoga_tree_t b, gutoga_stats_t *stats, double *prob)
{
	double cur = a.score;
	if (b.score > cur) {
		cur = b.score;
	}
	if (cur < stats->avg) {
		*prob = 1;
	}
	else {
		*prob = ((stats->max - cur) / (stats->max - stats->avg));
	}
	return GUTOGA_SUCCESS;
}

int gutoga_mutation_flip(void *context, gutoga_tree_t *t, gutoga_random_int_fn rnd)
{
	int pos = rnd(context, t->size - 2);
	memset(t->nodes, 0, gutoga_tree_num_nodes(t->size) * sizeof(*t->nodes));

	gutoga_tree_default(t);
	t->enc[pos] = rnd(context, 2 * pos + 3);

	return gutoga_tree_sequence(t);
}

int gutoga_population_crossover(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats)
{
	double start = omp_get_wtime();
	gutoga_population_reset(context, ga, ga->pop);
	gutoga_population_shuffle(context, ga->mates, ga->cfg->rand_int);

	for (int i = 0; i < ga->mates->size / 2; i++) {
		gutoga_tree_t parent_a = ga->mates->trees[2 * i];
		gutoga_tree_t parent_b = ga->mates->trees[2 * i + 1];

		gutoga_tree_t child_a;
		gutoga_tree_t child_b;
		double prob = 0.0;

		int rc = ga->cfg->crossover_prob(context, parent_a, parent_b, stats, &prob);
		if (rc != 0) {
			return rc;
		}
		rc = ga->cfg->crossover(context, parent_a, parent_b, &child_a, &child_b, ga->cfg->rand_int);
		if (rc != 0) {
			return rc;
		}

		rc = gutoga_population_append(ga->pop, child_a);
		if (rc != 0) {
			return rc;
		}

		rc = gutoga_population_append(ga->pop, child_b);
		if (rc != 0) {
			return rc;
		}
	}
	if (ga->mates->size % 2 == 1) {
		gutoga_tree_t tree;
		int rc = gutoga_tree_clone(context, &tree, ga->mates->trees[ga->mates->size - 1]);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
		rc = gutoga_population_append(ga->mates, tree);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
	}
	ga->cfg->logger(context, "crossover elapsed: %f; (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf)\n", omp_get_wtime() - start, stats->min, stats->max, stats->avg, stats->best.score);
	return GUTOGA_SUCCESS;
}

int gutoga_population_mutation(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats)
{
	double start = omp_get_wtime();
	for (int i = 0; i < ga->pop->size; i++) {
		double prob = 0.0;
		int rc = ga->cfg->mutation_prob(context, ga->pop->trees[i], stats, &prob);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
		if (ga->cfg->rand_double(context) <= prob) {
			continue;
		}
		rc = ga->cfg->mutation(context, ga->pop->trees + i, ga->cfg->rand_int);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
		ga->pop->trees[i].has_score = false;
		ga->pop->trees[i].score = 0;
	}
	ga->cfg->logger(context, "mutation elapsed: %f; (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf)\n", omp_get_wtime() - start, stats->min, stats->max, stats->avg, stats->best.score);
	return GUTOGA_SUCCESS;
}

int gutoga_population_fitness(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats)
{
	double start = omp_get_wtime();
#pragma omp parallel
	{
		omp_set_num_threads(ga->cfg->nthreads);
		int failed = 0;
#pragma omp for schedule(dynamic)
		for (int i = 0; i < ga->pop->size; i++) {
			int rc = GUTOGA_SUCCESS;
#pragma omp atomic read
			rc = failed;

			if (rc != GUTOGA_SUCCESS && rc != GUTOGA_ERR_FITNESS_SKIPPABLE) {
				continue;
			}
			gutoga_node_hash(ga, ga->pop->trees[i].root);
			rc = gutoga_tree_fitness(context, ga, ga->pop->trees + i, NULL);
			if (rc != GUTOGA_SUCCESS && rc != GUTOGA_ERR_FITNESS_SKIPPABLE) {
#pragma omp atomic write
				failed = rc;
			}
		}
	}

	double sum = 0.0;
	int max_ind = -1;
	for (int i = 0; i < ga->pop->size; i++) {
		double fitness;
		int rc = gutoga_tree_fitness(context, ga, ga->pop->trees + i, &fitness);
		if (rc != GUTOGA_SUCCESS && rc != GUTOGA_ERR_FITNESS_SKIPPABLE) {
			return rc;
		}
		if (fitness < stats->min) {
			stats->min = fitness;
		}
		if (fitness > stats->max) {
			stats->max = fitness;
			max_ind = i;
		}
		sum += fitness;
	}

	stats->avg = sum / ga->pop->size;
	if (stats->best.size == 0) {
		int rc = gutoga_tree_clone(context, &stats->best, ga->pop->trees[max_ind]);
		ga->cfg->logger(context, "fitness elapsed: %f; (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf)\n", omp_get_wtime() - start, stats->min, stats->max, stats->avg, stats->best.score);
		return rc;
	}
	if (stats->best.score < stats->max) {
		gutoga_tree_destroy(context, ga, stats->best);
		int rc = gutoga_tree_clone(context, &stats->best, ga->pop->trees[max_ind]);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
	}
	ga->cfg->logger(context, "fitness elapsed: %f; (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf)\n", omp_get_wtime() - start, stats->min, stats->max, stats->avg, stats->best.score);
	return GUTOGA_SUCCESS;
}

int gutoga_run_iteration(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats)
{
	for (gutoga_iteration_step_fn *step = ga->cfg->iteration_steps; *step != NULL; step++) {
		int rc = (*step)(context, ga, stats);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
	}

	return GUTOGA_SUCCESS;
}

int gutoga_population_init(void *context, gutoga_ga_t *ga)
{
	for (int i = 0; i < ga->cfg->population_size; i++) {
		gutoga_tree_t t;
		int rc = gutoga_tree_random(context, &t, ga->cfg->tree_size, ga->cfg->rand_int);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
		rc = gutoga_population_append(ga->pop, t);
		if (rc != GUTOGA_SUCCESS) {
			return rc;
		}
	}
	return GUTOGA_SUCCESS;
}

int gutoga_cfg_verify(gutoga_cfg_t *cfg)
{
	if (cfg->population_size < 2) {
		return GUTOGA_ERR_CFG_POPULATION_SIZE;
	}
	if (cfg->tree_size < 3) {
		return GUTOGA_ERR_CFG_TREE_SIZE;
	}
	if (cfg->nthreads < 1) {
		return GUTOGA_ERR_CFG_NTHREADS;
	}
	if (cfg->rand_int == NULL) {
		return GUTOGA_ERR_CFG_RAND_INT;
	}
	if (cfg->rand_double == NULL) {
		return GUTOGA_ERR_CFG_RAND_DOUBLE;
	}
	if (cfg->mutation_prob == NULL) {
		return GUTOGA_ERR_CFG_MUTATION_PROB;
	}
	if (cfg->mutation == NULL) {
		return GUTOGA_ERR_CFG_MUTATION;
	}
	if (cfg->crossover_prob == NULL) {
		return GUTOGA_ERR_CFG_CROSSOVER_PROB;
	}
	if (cfg->crossover == NULL) {
		return GUTOGA_ERR_CFG_CROSSOVER;
	}
	if (cfg->id_to_ptr == NULL) {
		return GUTOGA_ERR_CFG_ID_TO_PTR;
	}
	if (cfg->align == NULL) {
		return GUTOGA_ERR_CFG_ALIGN;
	}
	if (cfg->cleanup == NULL) {
		return GUTOGA_ERR_CFG_CLEANUP;
	}
	if (cfg->population_init == NULL) {
		return GUTOGA_ERR_CFG_POPULATION_INIT;
	}
	if (cfg->iteration_steps == NULL) {
		return GUTOGA_ERR_CFG_ITERATION_STEPS;
	}
	if (cfg->termination_criteria == NULL) {
		return GUTOGA_ERR_CFG_TERMINATION_CRITERIA;
	}
	if (cfg->logger == NULL) {
		return GUTOGA_ERR_CFG_LOGGER;
	}
	if (cfg->cache_ttl < 1) {
		return GUTOGA_ERR_CFG_CACHE_TTL;
	}
	return GUTOGA_SUCCESS;
}

int gutoga_ga_init(gutoga_ga_t *ga, gutoga_cfg_t *cfg)
{
	int rc = gutoga_cfg_verify(cfg);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}
	ga->cfg = cfg;

	ga->pop = gutoga_population_new(cfg->population_size);
	if (ga->pop == NULL) {
		return GUTOGA_ERR_BAD_ALLOC_POPULATION;
	}

	ga->mates = gutoga_population_new(cfg->population_size);
	if (ga->mates == NULL) {
		return GUTOGA_ERR_BAD_ALLOC_POPULATION;
	}
	ga->cache = gutoga_cache_new(cfg->cache_ttl);
	if (ga->cache == NULL) {
		return GUTOGA_ERR_BAD_ALLOC_CACHE;
	}
	return GUTOGA_SUCCESS;
}

void gutoga_ga_destroy(void *context, gutoga_ga_t *ga)
{
	gutoga_cache_destroy(context, ga->cache, ga->cfg->cleanup);
	gutoga_population_destroy(context, ga, ga->pop);
	gutoga_population_destroy(context, ga, ga->mates);
}

static double time_now()
{
	struct timespec tp;
	clock_gettime(CLOCK_MONOTONIC, &tp);
	return (double)tp.tv_sec + (double)tp.tv_nsec / 1000000000;
}

int gutoga_run_ga(void *context, gutoga_ga_t *ga, void **best)
{
	double ga_start = time_now();

	int rc = ga->cfg->population_init(context, ga);
	if (rc != GUTOGA_SUCCESS) {
		return rc;
	}

	gutoga_stats_t stats = {0};

	double elapsed = 0.0;
	bool terminate = false;
	double best_score = -1;
	for (stats.iteration = 0; rc == GUTOGA_SUCCESS && !terminate; stats.iteration++) {
		double start = time_now();
		stats.min = DBL_MAX;
		stats.max = -1;
		rc = gutoga_run_iteration(context, ga, &stats);
		elapsed = time_now() - start;
		if (rc != GUTOGA_SUCCESS) {
			break;
		}
		stats.elapsed = time_now() - ga_start;
		if (best_score >= stats.max) {
			stats.unchanged++;
		}
		else {
			best_score = stats.max;
			stats.unchanged = 0;
		}
		ga->cfg->logger(context, "ga: iteration %u: took %.5lfs (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf;unchanged=%d;elapsed=%.5lfs)\n",
				stats.iteration, elapsed, stats.min, stats.max, stats.avg, stats.best.score, stats.unchanged, stats.elapsed);
		rc = ga->cfg->termination_criteria(context, ga, &stats, &terminate);
		gutoga_cache_epoch(context, ga->cache, ga->cfg->cleanup);
	}
	if (rc != GUTOGA_SUCCESS) {
		ga->cfg->logger(context, "ga: iteration %u: elapsed %.5lfs (fail, code: %d)\n", stats.iteration, elapsed, rc);
		return rc;
	}
	ga->cfg->logger(context, "ga: iteration %u: took %.5lfs (min=%.5lf;max=%.5lf;avg=%.5lf;best=%.5lf;unchanged=%d;elapsed=%.5lfs)\n",
			stats.iteration, elapsed, stats.min, stats.max, stats.avg, stats.best.score, stats.unchanged, stats.elapsed);

	rc = gutoga_tree_force_fitness(context, ga, &stats.best, best);
	gutoga_tree_destroy(context, ga, stats.best);
	return rc;
}
