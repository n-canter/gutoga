#ifndef GUTOGA_H
#define GUTOGA_H

#include <stdbool.h>
#include <stdint.h>

#include "sha256.h"
#include "uthash.h"

struct gutoga_cache_entry_t {
	BYTE hash[SHA256_BLOCK_SIZE];
	int ttl;
	int refs;

	void *ptr;
	double score;

	UT_hash_handle hh;
};

struct gutoga_mutex_t {
	omp_lock_t l;
};

typedef struct gutoga_cache_t {
	int ttl;
	struct gutoga_mutex_t *lock;
	struct gutoga_cache_entry_t *records;
} gutoga_cache_t;

enum {
	GUTOGA_SUCCESS = 0,
	GUTOGA_ERR_BAD_TREE_SIZE = -100,

	GUTOGA_ERR_BAD_ALLOC = -200,
	GUTOGA_ERR_BAD_ALLOC_TREE = -201,
	GUTOGA_ERR_BAD_ALLOC_POPULATION = -202,
	GUTOGA_ERR_BAD_ALLOC_SELECTION = -203,
	GUTOGA_ERR_BAD_ALLOC_ORDER = -204,
	GUTOGA_ERR_BAD_ALLOC_CACHE = -205,

	GUTOGA_ERR_CROSSOVER = -400,
	GUTOGA_ERR_BAD_TREE = -401,
	GUTOGA_ERR_FITNESS_SKIPPABLE = -501,

	GUTOGA_ERR_CFG_POPULATION_SIZE = -600,
	GUTOGA_ERR_CFG_TREE_SIZE = -601,
	GUTOGA_ERR_CFG_NTHREADS = -602,
	GUTOGA_ERR_CFG_RAND_INT = -603,
	GUTOGA_ERR_CFG_RAND_DOUBLE = -604,
	GUTOGA_ERR_CFG_MUTATION_PROB = -605,
	GUTOGA_ERR_CFG_MUTATION = -606,
	GUTOGA_ERR_CFG_CROSSOVER_PROB = -607,
	GUTOGA_ERR_CFG_CROSSOVER = -608,
	GUTOGA_ERR_CFG_ID_TO_PTR = -609,
	GUTOGA_ERR_CFG_ALIGN = -610,
	GUTOGA_ERR_CFG_CLEANUP = -611,
	GUTOGA_ERR_CFG_POPULATION_INIT = -612,
	GUTOGA_ERR_CFG_ITERATION_STEPS = -613,
	GUTOGA_ERR_CFG_TERMINATION_CRITERIA = -614,
	GUTOGA_ERR_CFG_LOGGER = -615,
	GUTOGA_ERR_CFG_CACHE_TTL = -616,
};


typedef int (*gutoga_random_int_fn)(void *context, const int n);
typedef double (*gutoga_random_double_fn)(void *context);

typedef struct gutoga_cfg_t gutoga_cfg_t;

typedef struct gutoga_node_t {
	struct gutoga_node_t *left;
	struct gutoga_node_t *right;
	BYTE hash[SHA256_BLOCK_SIZE];

	int id;
	int left_edges;
	int right_edges;
} gutoga_node_t;

typedef struct gutoga_tree_t {
	int size;

	int *enc;
	gutoga_node_t *root;
	gutoga_node_t *nodes;
	int nodes_len;

	double score;
	bool has_score;
} gutoga_tree_t;

typedef struct gutoga_population_t {
	int size;
	int cap;

	gutoga_tree_t *trees;
} gutoga_population_t;

typedef struct gutoga_ga_t {
	gutoga_cfg_t *cfg;
	gutoga_population_t *pop;
	gutoga_population_t *mates;
	gutoga_cache_t *cache;
} gutoga_ga_t;

typedef struct gutoga_stats_t {
	int iteration;
	int unchanged;
	double elapsed;

	double min;
	double max;
	double avg;

	gutoga_tree_t best;
} gutoga_stats_t;


typedef int (*gutoga_mutation_prob_fn)
    (void *context,
     const gutoga_tree_t t, gutoga_stats_t *stats, double *prob);

typedef int (*gutoga_mutation_fn)
    (void *context,
     gutoga_tree_t *t, gutoga_random_int_fn rnd);

typedef int (*gutoga_crossover_prob_fn)
    (void *context,
     const gutoga_tree_t a, const gutoga_tree_t b,
     gutoga_stats_t *stats, double *prob);

typedef int (*gutoga_crossover_fn)
    (void *context,
     const gutoga_tree_t parent_a, const gutoga_tree_t parent_b,
     gutoga_tree_t *child_a, gutoga_tree_t *child_b,
     gutoga_random_int_fn rnd);

typedef int (*gutoga_ga_population_init_fn)
    (void *context, gutoga_ga_t *ga);

typedef int (*gutoga_termination_criteria_fn)
    (void *context, gutoga_ga_t *ga, gutoga_stats_t *stats, bool *terminate);

typedef int (*gutoga_iteration_step_fn)
    (void *context, gutoga_ga_t *ga, gutoga_stats_t *stats);

typedef int (*gutoga_id_to_ptr_fn)
    (void *context, const int id, void **data);

typedef int (*gutoga_align_fn)
    (void *context, void *s1, void *s2, void **data, double *score);

typedef void (*gutoga_cleanup_fn)
    (void *context, void *data);

typedef int (*gutoga_logger_fn)
    (void *context, const char *fmt, ...);

struct gutoga_cfg_t {
	int population_size;
	int tree_size;

	int nthreads;

	int cache_ttl;

	gutoga_random_int_fn rand_int;
	gutoga_random_double_fn rand_double;

	gutoga_mutation_prob_fn mutation_prob;
	gutoga_mutation_fn mutation;

	gutoga_crossover_prob_fn crossover_prob;
	gutoga_crossover_fn crossover;

	gutoga_id_to_ptr_fn id_to_ptr;
	gutoga_align_fn align;
	gutoga_cleanup_fn cleanup;

	gutoga_ga_population_init_fn population_init;

	gutoga_iteration_step_fn *iteration_steps;
	gutoga_termination_criteria_fn termination_criteria;

	gutoga_logger_fn logger;
};

int gutoga_selection_roulette_wheel(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats);
int gutoga_population_crossover(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats);
int gutoga_population_mutation(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats);
int gutoga_population_fitness(void *context, gutoga_ga_t *ga, gutoga_stats_t *stats);
void gutoga_tree_destroy(void *context, gutoga_ga_t *ga, gutoga_tree_t t);
void gutoga_ga_destroy(void *context, gutoga_ga_t *ga);
int gutoga_run_ga(void *context, gutoga_ga_t *ga, void **best);
int gutoga_population_init(void *context, gutoga_ga_t *ga);
int gutoga_mutation_flip(void *context, gutoga_tree_t *t, gutoga_random_int_fn rnd);
int gutoga_crossover_one_point(void *context,
			       const gutoga_tree_t parent_a, const gutoga_tree_t parent_b,
			       gutoga_tree_t *child_a, gutoga_tree_t *child_b,
			       gutoga_random_int_fn rnd);
int gutoga_ga_init(gutoga_ga_t *ga, gutoga_cfg_t *cfg);
int gutoga_mutation_adaptive_prob(void *context, const gutoga_tree_t t, gutoga_stats_t *stats, double *prob);
int gutoga_crossover_adaptive_prob(void *context, const gutoga_tree_t a, const gutoga_tree_t b, gutoga_stats_t *stats, double *prob);

#endif
