#ifndef TREENODE_H
#define TREENODE_H

#include <memory>

class TreeNode
{
public:
    // constructors
    TreeNode(bool _is_leaf) : depth(0), split_attr(-1), split_value(-1), split_gain(-1), is_leaf(_is_leaf){};

    // fields
    std::shared_ptr<TreeNode> left, right;
    int depth;
    int split_attr;
    double split_value;
    double split_gain;
    int lhs_size, rhs_size;
    bool is_leaf;
    double prediction; // if it's a leaf
};

#endif // TREENODE_H