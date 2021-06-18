#ifndef TREENODE_H
#define TREENODE_H

#include "utils.h"

class TreeNode {
public:
    TreeNode(bool is_leaf);
    ~TreeNode();

    // probably need some info/pointers to dataset

    //void print_node_info();

    bool is_leaf();
    TreeNode *left, *right;
    int depth;
    int split_attr;
    float split_value;
    float split_gain;
    float weight;
    float prediction; // if it's a leaf
};


#endif // TREENODE_H