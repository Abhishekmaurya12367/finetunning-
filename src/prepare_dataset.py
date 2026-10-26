
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Each topic contains an explanation, complexity, use case, and example.
TOPICS = {
    "Array": {
        "definition": "An array stores elements in contiguous indexed positions.",
        "complexity": "Index access is O(1), while searching is O(n) in an unsorted array.",
        "use": "Arrays are useful for storing ordered collections and performing index-based access.",
        "example": "In [10, 20, 30], the element at index 1 is 20."
    },
    "Linked List": {
        "definition": "A linked list consists of nodes that store data and links to other nodes.",
        "complexity": "Searching is O(n), and insertion at the head is O(1).",
        "use": "Linked lists are useful when frequent insertions and deletions are needed.",
        "example": "A list can contain nodes 10 -> 20 -> 30."
    },
    "Stack": {
        "definition": "A stack follows LIFO, meaning Last In, First Out.",
        "complexity": "Push, pop, and top are O(1) for a typical stack implementation.",
        "use": "Stacks are useful for recursion, undo operations, and balanced parentheses.",
        "example": "Push 10, then push 20. Popping returns 20 first."
    },
    "Queue": {
        "definition": "A queue follows FIFO, meaning First In, First Out.",
        "complexity": "Enqueue and dequeue can both be O(1) with a suitable implementation.",
        "use": "Queues are used in BFS, scheduling, and task processing.",
        "example": "Enqueue 10 and then 20. Dequeue returns 10 first."
    },
    "Deque": {
        "definition": "A deque is a double-ended queue that supports insertion and deletion at both ends.",
        "complexity": "Insertion and deletion at either end are typically O(1).",
        "use": "Deques are useful for sliding-window maximum problems.",
        "example": "A deque can add 10 at the front and 20 at the back."
    },
    "Hash Map": {
        "definition": "A hash map stores key-value pairs using a hash function.",
        "complexity": "Lookup, insertion, and deletion are O(1) on average and can degrade to O(n).",
        "use": "Hash maps are useful for frequency counting, caching, and fast lookups.",
        "example": "A map can store the key 'apple' with the value 3."
    },
    "Hash Set": {
        "definition": "A hash set stores unique values using hashing.",
        "complexity": "Insertion, deletion, and membership checks are O(1) on average.",
        "use": "Hash sets are useful for duplicate detection and membership testing.",
        "example": "Adding 5 twice still leaves only one copy of 5."
    },
    "Binary Search": {
        "definition": "Binary search repeatedly halves a sorted search interval.",
        "complexity": "Its time complexity is O(log n) on a sorted array.",
        "use": "It is useful for searching sorted data and finding monotonic boundaries.",
        "example": "Searching for 7 in [1, 3, 5, 7, 9] finds the element at index 3."
    },
    "Two Pointers": {
        "definition": "The two-pointer technique uses two indices to process a collection efficiently.",
        "complexity": "Many two-pointer algorithms run in O(n) time.",
        "use": "It is useful for sorted pair-sum problems and removing duplicates.",
        "example": "Pointers can move inward from both ends of a sorted array."
    },
    "Sliding Window": {
        "definition": "A sliding window maintains a range of elements while moving through a sequence.",
        "complexity": "Many sliding-window algorithms run in O(n) time.",
        "use": "It is useful for subarray sums and substring problems.",
        "example": "A window of size 3 moves across [1, 2, 3, 4, 5]."
    },
    "Prefix Sum": {
        "definition": "A prefix sum stores cumulative sums of array elements.",
        "complexity": "Building prefix sums takes O(n), and range-sum queries take O(1).",
        "use": "It is useful for answering repeated range-sum queries.",
        "example": "For [2, 3, 4], prefix sums are [2, 5, 9]."
    },
    "Difference Array": {
        "definition": "A difference array represents changes between consecutive array values.",
        "complexity": "Range updates can take O(1) each, followed by O(n) reconstruction.",
        "use": "It is useful for applying many range increment operations.",
        "example": "An update adding 5 to indices 1 through 3 can be represented by two boundary changes."
    },
    "Sorting": {
        "definition": "Sorting arranges elements according to a specified order.",
        "complexity": "Merge sort and heap sort take O(n log n); quicksort has O(n log n) average time.",
        "use": "Sorting helps with searching, ranking, and duplicate detection.",
        "example": "Sorting [3, 1, 2] in ascending order produces [1, 2, 3]."
    },
    "Merge Sort": {
        "definition": "Merge sort divides an array, sorts the parts, and merges them.",
        "complexity": "Its time complexity is O(n log n), with O(n) auxiliary space for arrays.",
        "use": "It is useful when predictable sorting performance is required.",
        "example": "The arrays [1, 3] and [2, 4] merge into [1, 2, 3, 4]."
    },
    "Quick Sort": {
        "definition": "Quicksort partitions elements around a pivot and recursively sorts the partitions.",
        "complexity": "Its average time is O(n log n), while its worst case is O(n^2).",
        "use": "It is useful for general-purpose in-memory sorting.",
        "example": "Partitioning [3, 1, 2] around a pivot rearranges elements by their relation to the pivot."
    },
    "Heap": {
        "definition": "A heap is a complete binary tree that satisfies a heap-order property.",
        "complexity": "Getting the root is O(1), and insertion and removal are O(log n).",
        "use": "Heaps are useful for priority queues and top-k problems.",
        "example": "A min-heap containing 3, 5, and 8 has 3 at its root."
    },
    "Monotonic Stack": {
        "definition": "A monotonic stack maintains elements in increasing or decreasing order.",
        "complexity": "Many monotonic-stack algorithms run in O(n) because each element is pushed and popped at most once.",
        "use": "It is useful for next-greater-element and histogram-area problems.",
        "example": "A decreasing stack can help find the next greater element for each array position."
    },
    "Recursion": {
        "definition": "Recursion solves a problem by calling a function on smaller instances of the same problem.",
        "complexity": "Time and space complexity depend on the recurrence and recursion depth.",
        "use": "It is useful for tree traversal, divide-and-conquer, and backtracking.",
        "example": "Factorial satisfies factorial(n) = n * factorial(n - 1), with a base case at zero."
    },
    "Backtracking": {
        "definition": "Backtracking explores candidate solutions and reverses choices when a path cannot produce a valid solution.",
        "complexity": "Its worst-case time is often exponential.",
        "use": "It is useful for permutations, combinations, Sudoku, and N-Queens.",
        "example": "A permutation algorithm chooses one unused element, recurses, and then removes that choice."
    },
    "Binary Tree": {
        "definition": "A binary tree is a tree in which each node has at most two children.",
        "complexity": "Traversing all nodes takes O(n) time.",
        "use": "Binary trees are useful for hierarchical data and recursive algorithms.",
        "example": "A node with value 1 can have left child 2 and right child 3."
    },
    "Binary Search Tree": {
        "definition": "A binary search tree orders keys so that left-subtree keys are smaller and right-subtree keys are larger, under the usual distinct-key convention.",
        "complexity": "Search, insertion, and deletion take O(h), where h is the tree height.",
        "use": "BSTs support ordered searching and traversal.",
        "example": "In a BST rooted at 8, values 4 and 12 can be placed on the left and right."
    },
    "AVL Tree": {
        "definition": "An AVL tree is a self-balancing binary search tree.",
        "complexity": "Search, insertion, and deletion take O(log n).",
        "use": "It is useful when balanced search-tree operations are required.",
        "example": "AVL rotations restore balance after an insertion makes a node unbalanced."
    },
    "Trie": {
        "definition": "A trie stores strings through shared prefixes in a tree-like structure.",
        "complexity": "Searching for a word takes O(L), where L is the word length, ignoring implementation-specific costs.",
        "use": "Tries are useful for autocomplete, dictionaries, and prefix searches.",
        "example": "The words 'car' and 'cat' share the prefix 'ca'."
    },
    "Tree Traversal": {
        "definition": "Tree traversal visits the nodes of a tree in a specified order.",
        "complexity": "Standard traversals take O(n) time.",
        "use": "Traversals are useful for processing hierarchical structures.",
        "example": "Inorder traversal visits left subtree, root, and right subtree."
    },
    "Lowest Common Ancestor": {
        "definition": "The lowest common ancestor is the deepest node that is an ancestor of two given nodes.",
        "complexity": "A standard binary-tree DFS solution takes O(n) time.",
        "use": "It is useful for finding shared ancestry in trees.",
        "example": "If nodes 4 and 5 are children of 2, their lowest common ancestor is 2."
    },
    "Graph": {
        "definition": "A graph consists of vertices connected by edges.",
        "complexity": "Graph algorithm complexity depends on the representation and the operation.",
        "use": "Graphs model networks, dependencies, routes, and relationships.",
        "example": "Vertices A and B connected by an edge form a simple graph."
    },
    "BFS": {
        "definition": "Breadth-First Search explores a graph level by level using a queue.",
        "complexity": "With an adjacency list, BFS takes O(V + E) time.",
        "use": "It is useful for shortest paths in unweighted graphs.",
        "example": "Starting at A, BFS visits A's immediate neighbors before exploring the next level."
    },
    "DFS": {
        "definition": "Depth-First Search explores a path as far as possible before backtracking.",
        "complexity": "With an adjacency list, DFS takes O(V + E) time.",
        "use": "It is useful for connectivity, cycle detection, and graph traversal.",
        "example": "DFS visits a neighbor and recursively explores that neighbor's unvisited neighbors."
    },
    "Topological Sort": {
        "definition": "Topological sorting orders the vertices of a directed acyclic graph so every edge goes from earlier to later.",
        "complexity": "It takes O(V + E) time with standard algorithms.",
        "use": "It is useful for course prerequisites and build dependencies.",
        "example": "If A must happen before B, a topological order places A before B."
    },
    "Dijkstra Algorithm": {
        "definition": "Dijkstra's algorithm computes shortest paths from a source when edge weights are nonnegative.",
        "complexity": "With an adjacency list and binary heap, it typically takes O((V + E) log V).",
        "use": "It is useful for shortest-route problems with nonnegative weights.",
        "example": "If A-B costs 2 and B-C costs 3, the path A-B-C costs 5."
    },
    "Bellman-Ford": {
        "definition": "Bellman-Ford computes single-source shortest paths and can detect reachable negative-weight cycles.",
        "complexity": "Its time complexity is O(VE).",
        "use": "It is useful for graphs that may contain negative edge weights.",
        "example": "It repeatedly relaxes all edges up to V - 1 times."
    },
    "Minimum Spanning Tree": {
        "definition": "A minimum spanning tree connects all vertices of a connected weighted undirected graph with minimum total edge weight and no cycles.",
        "complexity": "Kruskal's algorithm typically takes O(E log E).",
        "use": "It is useful for designing low-cost networks.",
        "example": "Connecting four cities with minimum total road cost can be modeled as an MST problem."
    },
    "Disjoint Set Union": {
        "definition": "Disjoint Set Union maintains groups of elements using find and union operations.",
        "complexity": "With path compression and union by rank or size, operations are nearly O(1) amortized.",
        "use": "It is useful for connectivity queries and Kruskal's algorithm.",
        "example": "Union(1, 2) places elements 1 and 2 in the same component."
    },
    "Dynamic Programming": {
        "definition": "Dynamic programming solves overlapping subproblems and stores their results to avoid repeated computation.",
        "complexity": "Complexity depends on the number of states and transitions.",
        "use": "It is useful for optimization and counting problems with suitable recurrence relations.",
        "example": "Fibonacci values can be stored so each value is computed only once."
    },
    "Memoization": {
        "definition": "Memoization stores the results of previously solved recursive subproblems.",
        "complexity": "It can reduce exponential recursive algorithms to polynomial time when there are only polynomially many states and transitions.",
        "use": "It is useful for top-down dynamic programming.",
        "example": "A cached Fibonacci function reuses the computed value of fib(5)."
    },
    "Tabulation": {
        "definition": "Tabulation computes dynamic-programming states iteratively in dependency order.",
        "complexity": "Its time complexity depends on the number of states and transitions.",
        "use": "It is useful for bottom-up dynamic programming without recursive call overhead.",
        "example": "Fibonacci tabulation starts with F(0) and F(1), then computes later values."
    },
    "Knapsack": {
        "definition": "The 0/1 knapsack problem chooses items with weights and values under a capacity constraint, using each item at most once.",
        "complexity": "The standard dynamic-programming solution takes O(nW) time, where W is capacity.",
        "use": "It is useful for resource-allocation optimization.",
        "example": "A capacity of 5 cannot accommodate items whose combined weight is 7."
    },
    "Greedy Algorithm": {
        "definition": "A greedy algorithm makes a locally optimal choice at each step.",
        "complexity": "Complexity depends on the choices and any sorting or data structures used.",
        "use": "It is useful when a problem has the greedy-choice property and optimal substructure.",
        "example": "Kruskal's algorithm repeatedly selects the lowest-weight edge that does not create a cycle."
    },
    "Bit Manipulation": {
        "definition": "Bit manipulation operates directly on the binary representation of integers.",
        "complexity": "Basic bitwise operations take O(1) time for fixed-width integers.",
        "use": "It is useful for masks, flags, parity checks, and efficient state representation.",
        "example": "For an integer x, x & 1 checks whether its least significant bit is set."
    },
    "Kadane Algorithm": {
        "definition": "Kadane's algorithm finds the maximum-sum contiguous subarray in a one-dimensional array.",
        "complexity": "Its time complexity is O(n) and auxiliary space can be O(1).",
        "use": "It is useful for maximum subarray sum problems.",
        "example": "For [-2, 1, -3, 4, -1, 2], the maximum subarray is [4, -1, 2] with sum 5."
    },
    "Segment Tree": {
        "definition": "A segment tree stores information about array intervals in a tree structure.",
        "complexity": "Building takes O(n), while typical point updates and range queries take O(log n).",
        "use": "It is useful for range-sum, minimum, maximum, and other associative range queries.",
        "example": "A segment tree can answer the sum of array elements from index 2 to index 5."
    },
    "Fenwick Tree": {
        "definition": "A Fenwick tree, also called a Binary Indexed Tree, maintains prefix aggregates efficiently.",
        "complexity": "Updates and prefix queries take O(log n).",
        "use": "It is useful for dynamic prefix sums and frequency counting.",
        "example": "A Fenwick tree can update one array position and then query the sum of the first five elements."
    },
}


def build_examples():
    examples = []

    for topic, info in TOPICS.items():
        question_answer_pairs = [
            (
                f"What is {topic}?",
                info["definition"]
            ),
            (
                f"Explain {topic} in simple words.",
                info["definition"] + " " + info["use"]
            ),
            (
                f"What is the time complexity associated with {topic}?",
                info["complexity"]
            ),
            (
                f"When should we use {topic}?",
                info["use"]
            ),
            (
                f"Give a simple example of {topic}.",
                info["example"]
            ),
            (
                f"Explain the main idea behind {topic}.",
                info["definition"] + " " + info["use"]
            ),
            (
                f"What should a beginner know about {topic}?",
                info["definition"] + " " + info["complexity"]
            ),
            (
                f"Give an interview-style explanation of {topic}.",
                info["definition"] + " " + info["complexity"] + " " + info["use"]
            ),
        ]

        for question, answer in question_answer_pairs:
            examples.append({
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            })

    return examples


def save_jsonl(examples, filename):
    path = DATA_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Created: {path} ({len(examples)} examples)")


if __name__ == "__main__":
    examples = build_examples()

    # Reproducible shuffle
    random.Random(42).shuffle(examples)

    # Split the examples into training and validation sets
    split_index = int(len(examples) * 0.9)
    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]

    save_jsonl(train_examples, "train.jsonl")
    save_jsonl(validation_examples, "validation.jsonl")

    print(f"Total examples: {len(examples)}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(validation_examples)}")
