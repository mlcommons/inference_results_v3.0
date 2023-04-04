#ifndef ITEM_H_
#define ITEM_H_

#include <vector>
#include <loadgen.h>
#include <query_sample.h>
#include <query_sample_library.h>

struct Item {

    // For incoming query items from loadgen
    mlperf::ResponseId response_id_; 
    mlperf::QuerySampleIndex sample_idx_;

    // Will be used for when fetching or processing samples (fetching and post-processing)
	std::vector<mlperf::ResponseId> response_ids_; 
	std::vector<mlperf::QuerySampleIndex> sample_idxs_; 

    int number_dummies=0;

	Item(){};
	Item( std::vector<mlperf::ResponseId> response_ids, std::vector<mlperf::QuerySampleIndex> sample_idxs) :
           response_ids_(response_ids), sample_idxs_(sample_idxs) {}

	Item( std::vector<mlperf::ResponseId> response_ids, std::vector<mlperf::QuerySampleIndex> sample_idxs, int num_dummies) :
           response_ids_(response_ids), sample_idxs_(sample_idxs), number_dummies(num_dummies) {}

  
    Item( mlperf::ResponseId response_id, mlperf::QuerySampleIndex sample_idx) :
           response_id_(response_id), sample_idx_(sample_idx) {}
    
    Item( mlperf::ResponseId response_id, mlperf::QuerySampleIndex sample_idx,int n_dummies) :
           response_id_(response_id), sample_idx_(sample_idx), number_dummies(n_dummies) {}
}; // Item

#endif