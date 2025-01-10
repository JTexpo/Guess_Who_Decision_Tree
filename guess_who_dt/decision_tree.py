from typing import Optional, Tuple, List
import csv
import numpy as np


class TreeNode:
    def __init__(
        self,
        left_node: Optional["TreeNode"] = None,
        right_node: Optional["TreeNode"] = None,
        feature_value_index: int = None,
        feature_value_threshold_lte: float = None,
        variance_reduction: float = None,
        leaf_value: float = None,
    ):
        self.left_node = left_node
        self.right_node = right_node
        self.feature_value_index = feature_value_index
        self.feature_value_threshold_lte = feature_value_threshold_lte
        self.variance_reduction = variance_reduction
        self.leaf_value = leaf_value


class DecisionTree:
    def __init__(
        self,
        min_split_length: int = 2,
        max_depth: int = 2,
    ):
        self.root: TreeNode = None

        self.min_split_length = min_split_length
        self.max_depth = max_depth

def _get_tree_split(dataset: np.array) -> Tuple[float, int, float]:
    """The goal of this function is to find the index that has the most even number 
    of nodes on both the left and right. Finding this midpoint, helps us then find the
    values we're looking for with less recursion! 

    Args:
        dataset (np.array): 

    Returns:
        Tuple[float, int, float]: 
            best_variance_reduction :
            best_feature_index :
            best_0.5 :
    """
    _, dataset_input_features = np.shape(dataset[:, 1:])

    best_variance_reduction = float("-inf")
    best_feature_index = -1

    # itterating over all of the columns to find the best split in the data
    for feature_index in range(dataset_input_features):

        left_dataset = np.array(
            [
                value
                for value in dataset
                if value[feature_index + 1] <= 0.5
            ]
        )
        right_dataset = np.array(
            [
                value
                for value in dataset
                if value[feature_index + 1] > 0.5
            ]
        )

        if (len(left_dataset) <= 0) or (len(right_dataset) <= 0):
            continue
        
        # Getting the average amount of data both sides
        feature_variance_reduction = (len(dataset) - abs(len(left_dataset) - len(right_dataset)))/2.0
        
        if feature_variance_reduction < best_variance_reduction:
            continue

        best_variance_reduction = feature_variance_reduction
        best_feature_index = feature_index + 1

    return (
        best_variance_reduction,
        best_feature_index,
    )

def build_tree(
    dataset: np.array,
    depth: int = 0,
    min_split_length: int = 2,
    max_depth: int = 2,
) -> TreeNode:
    """This method will recursively build the decision tree, starting with the root

    Args:
        dataset (np.array): the dataset to build the tree on
        depth (int, optional): how deep we are in the tree. Defaults to 0.
        min_split_length (int, optional): how small we allow for a split. Defaults to 2.
        max_depth (int, optional): how deep we are allowed to go (this is a stopping condition). Defaults to 2.

    Returns:
        TreeNode: _description_
    """
    # columns 2+ are values
    dataset_inputs = dataset[:,1:]
    # column 1 is ID / Value
    # NOTE : Just as the string labels for the columns are cut off, 
    # I have also cut off the string column, and will instead use an ID to get that value
    dataset_outputs = dataset[:, 0]

    # if we can't split the tree any further or we have reached the max depth, than we end the recursion
    if (len(dataset_inputs) < min_split_length) or (depth > max_depth):
        return TreeNode(leaf_value=np.mean(dataset_outputs))
    
    # Finding the best split for the tree. This will result in the most even distribution of nodes on either branch
    (
        best_variance_reduction,
        best_feature_index,
    ) = _get_tree_split(dataset=dataset)

    # splitting up the data for the rules that were returned
    # ------------------------------------------------------
    left_dataset = np.array(
        [
            value
            for value in dataset
            if value[best_feature_index] <= 0.5
        ]
    )
    right_dataset = np.array(
        [
            value
            for value in dataset
            if value[best_feature_index] > 0.5
        ]
    )

    # Recursively building the left and right branches
    # ------------------------------------------------
    left_tree = build_tree(
        dataset=left_dataset,
        depth=depth + 1,
        min_split_length=min_split_length,
        max_depth=max_depth,
    )

    right_tree = build_tree(
        dataset=right_dataset,
        depth=depth + 1,
        min_split_length=min_split_length,
        max_depth=max_depth,
    )

    # Returning the tree once all done 
    return TreeNode(
        left_node=left_tree,
        right_node=right_tree,
        feature_value_index=best_feature_index,
        feature_value_threshold_lte=0.5,
        variance_reduction=best_variance_reduction,
    )


def print_tree(tree: TreeNode = None, indent="  "):
    """A recursive method to print the tree

    Args:
        tree (TreeNode, optional): WHEN FIRST CALLING THIS SHOULD BE None !!!! Defaults to None.
        indent (str, optional): The spacing to put before the stringed value. Defaults to "  ".
    """

    # if there is a leaf, then we reached the end of the tree and can return the leaf's value
    if tree.leaf_value is not None:
        print(tree.leaf_value)

    # printing the an indicator about which branch we are on and its threshold + variance reduction
    else:
        print(
            f"Dataset[{tree.feature_value_index}] <= {tree.feature_value_threshold_lte} ? ( Variance Reduction {tree.variance_reduction} )"
        )
        print(f"{indent}Left  : ", end="")
        print_tree(tree.left_node, indent + "  ")
        print(f"{indent}Right : ", end="")
        print_tree(tree.right_node, indent + "  ")