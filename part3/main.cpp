#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "alglibmisc.h"
#include <nlohmann/json.hpp>
#include <chrono>


using json = nlohmann::json;


int main(int argc, char* argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <query.json> <passages.json> <K> <eps>\n";
    return 1;
    }

    auto processing_start = std::chrono::high_resolution_clock::now();
    // Load and parse query JSON
    std::ifstream query_ifs(argv[1]);
    if (!query_ifs) {
        std::cerr << "Error opening query file: " << argv[1] << "\n";
        return 1;
    }
    json query_json;
    query_ifs >> query_json;
    if (!query_json.is_array() || query_json.size() < 1) {
        std::cerr << "Query JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Load and parse passages JSON
    std::ifstream passages_ifs(argv[2]);
    if (!passages_ifs) {
        std::cerr << "Error opening passages file: " << argv[2] << "\n";
        return 1;
    }
    json passages_json;
    passages_ifs >> passages_json;
    if (!passages_json.is_array() || passages_json.size() < 1) {
        std::cerr << "Passages JSON must be an array with at least 1 element\n";
        return 1;
    }


    // // Convert JSON array to a dict mapping id -> element
    // std::unordered_map<int, json> dict;
    // for (auto &elem : passages_json) {
    //     int id = elem["id"].get<int>();
    //     dict[id] = elem;
    // }


    // Parse K and eps
    int k = std::stoi(argv[3]);
    double eps = std::stof(argv[4]);

    try{
        // Extract the query embedding
        auto& query_obj   = query_json[0];
        size_t D         = query_obj["embedding"].size();
        alglib::real_1d_array query;
        query.setlength(D);
        for (size_t d = 0; d < D; ++d) {
            query[d] = query_obj["embedding"][d].get<double>();
        }
        /*
        TODO:
        1. Extract the passage embedding and store it in alglib::real_2d_array, store the idx of each embedding in alglib::integer_1d_array
        2. Build the KD-tree (alglib::kdtree) from the passages embeddings using alglib::buildkdtree
        3. Perform the k-NN search using alglib::knnsearch
        4. Query the results
            - Get the index of each found neighbour  using alglib::kdtreequeryresultstags
            - Get the distance between each found neighbour and the query embedding using alglib::kdtreequeryresultsdists
        */
        size_t num_embeddings = passages_json.size();
        alglib::real_2d_array embeddings;
        alglib::integer_1d_array ids;
        embeddings.setlength(num_embeddings, D);
        ids.setlength(num_embeddings);

        for (size_t e = 0; e < num_embeddings; ++e) {
            auto& embedding_obj = passages_json[e];
            for (size_t d = 0; d < D; ++d) {
                embeddings[e][d] = embedding_obj["embedding"][d].get<double>();
            }
            ids[e] = embedding_obj["id"].get<alglib::ae_int_t>();
        }
        auto processing_end = std::chrono::high_resolution_clock::now();

        auto tree_start = std::chrono::high_resolution_clock::now();
        alglib::kdtree tree;
        alglib::kdtreebuildtagged(embeddings, ids, num_embeddings, D, 0, 2, tree);
        auto tree_end = std::chrono::high_resolution_clock::now();

        auto knn_start = std::chrono::high_resolution_clock::now();
        alglib::ae_int_t num_neighbors_found = alglib::kdtreequeryaknn(tree, query, k, eps);
        alglib::real_1d_array found_dists;
        alglib::integer_1d_array found_ids;
        found_dists.setlength(num_neighbors_found);
        found_ids.setlength(num_neighbors_found);
        alglib::kdtreequeryresultsdistances(tree, found_dists);
        alglib::kdtreequeryresultstags(tree, found_ids);
        auto knn_end = std::chrono::high_resolution_clock::now();

        for (alglib::ae_int_t i = 0; i < num_neighbors_found; ++i) {
            std::cout << "Neighbor " << i+1 << ": distance = " << found_dists[i] << "; id = " << found_ids[i] << std::endl;
        }
    }
    catch(alglib::ap_error &e) {
        std::cerr << "ALGLIB error: " << e.msg << std::endl;
        return 1;
    }

    auto program_end = std::chrono::high_resolution_clock::now();

    std::cout << program_end - program_start << std::endl;

    return 0;
}