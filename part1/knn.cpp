#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T, int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    if (items.size() == 0) {
        return nullptr;
    }

    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    const size_t median_idx = (items.size() - 1) / 2;
    const std::pair<Embedding_T, int>& median = items[median_idx];

    Node* root = new Node();
    root->embedding = median.first;
    root->idx = median.second;
    std::vector left_items(items.begin(), items.begin() + median_idx);
    std::vector right_items(items.begin() + median_idx + 1, items.end());
    root->left = buildKD(left_items, depth + 1);
    root->right = buildKD(right_items, depth + 1);
    return root;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    // int k = 1; // change later
    // int axis = depth % k;

    if (K == 0 || node == nullptr) return;

    float distance =  Node::queryEmbedding - node->embedding;
    Node *near_subtree = distance < 0 ? node->left : node->right;
    Node *far_subtree = distance < 0 ? node->right : node->left;
    knnSearch(near_subtree, depth + 1, K, heap);

    float abs_distance = std::abs(distance);
    if (heap.size() <= K)
    {
        heap.emplace(abs_distance, node->idx);
    }
    else
    {
        float furthest = heap.top().first;
        if (abs_distance < furthest)
        {
            heap.pop();
            heap.emplace(abs_distance, node->idx);
        }
    }

    float new_furthest = heap.top().first;
    if (abs_distance < new_furthest)
    {
        knnSearch(far_subtree, depth + 1, K, heap);
    }
}