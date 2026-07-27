
# 1. What is Data Mining? (5 Marks)

**Definition:**
Data Mining is the process of extracting useful information, patterns, and knowledge from a large amount of data. It helps organizations make better decisions by analyzing hidden relationships in data.

### Key Points

* It is also called **Knowledge Discovery**.
* It extracts useful patterns from large datasets.
* It uses machine learning, statistics, and database techniques.
* It helps in prediction and decision-making.
* It is used in banking, healthcare, education, marketing, and fraud detection.

### Applications

* Loan approval
* Medical diagnosis
* Market analysis
* Customer segmentation
* Fraud detection

### Advantages

* Finds hidden patterns.
* Improves decision-making.
* Predicts future trends.
* Saves time and cost.

---

# 2. Difference between Data Mining and KDD (5 Marks)

**KDD (Knowledge Discovery in Databases)** is the complete process of discovering useful knowledge, while **Data Mining** is only one step in the KDD process.

| KDD                                       | Data Mining                 |
| ----------------------------------------- | --------------------------- |
| Complete process                          | One step of KDD             |
| Includes data cleaning and transformation | Extracts patterns from data |
| Produces useful knowledge                 | Produces patterns/models    |
| Starts with raw data                      | Uses prepared data          |

### Steps of KDD

1. Data Cleaning
2. Data Integration
3. Data Selection
4. Data Transformation
5. Data Mining
6. Pattern Evaluation
7. Knowledge Presentation

**Conclusion:** Data Mining is a part of the KDD process.

---

# 3. Supervised vs Unsupervised Learning (5 Marks)

Machine Learning is divided into **Supervised** and **Unsupervised** learning.

### Supervised Learning

* Uses labeled data.
* Output is already known.
* Used for classification and prediction.
* Examples: Decision Tree, SVM, OneR.

### Unsupervised Learning

* Uses unlabeled data.
* Output is unknown.
* Used for clustering.
* Example: K-Means.

| Supervised         | Unsupervised        |
| ------------------ | ------------------- |
| Uses labeled data  | Uses unlabeled data |
| Predicts classes   | Groups similar data |
| Decision Tree, SVM | K-Means             |

---

# 4. Classification vs Clustering (5 Marks)

### Classification

Classification is a supervised learning technique that assigns data to predefined classes.

**Example:** Good Credit / Bad Credit

### Clustering

Clustering is an unsupervised learning technique that groups similar data without predefined classes.

**Example:** Customer Segmentation

| Classification         | Clustering                      |
| ---------------------- | ------------------------------- |
| Supervised             | Unsupervised                    |
| Uses class labels      | No class labels                 |
| Predicts known classes | Forms groups of similar objects |
| Example: Decision Tree | Example: K-Means                |

---

# 5. What is a Decision Tree? (5 Marks)

A **Decision Tree** is a supervised machine learning algorithm used for classification and prediction. It represents decisions in the form of a tree structure.

### Components

* **Root Node:** Starting node.
* **Internal Node:** Represents a condition or test.
* **Branch:** Outcome of the condition.
* **Leaf Node:** Final decision or class.

### Advantages

* Easy to understand and interpret.
* Fast and simple to implement.
* Produces IF–THEN rules.
* Works well for classification problems.

### Disadvantages

* Can overfit the training data.
* Small changes in data may produce a different tree.

### Applications

* Loan approval
* Disease prediction
* Fraud detection
* Student performance analysis

---



---

# 6. What is Information Gain? (5 Marks)

### Definition

**Information Gain (IG)** is a measure used in Decision Trees to determine which attribute is the best for splitting the data. It measures how much an attribute reduces uncertainty or impurity.

### Key Points

* Used in Decision Tree algorithms like J48.
* Selects the most important attribute.
* Attribute with **highest Information Gain** becomes the root node.
* Helps create a better Decision Tree.
* Higher Information Gain means better classification.

### Formula

**Information Gain = Entropy (Parent) − Entropy (Children)**

### Example

Suppose a dataset has attributes:

* Age
* Income
* Checking Status

If **Checking Status** has the highest Information Gain, it becomes the root node.

### Advantages

* Selects the best attribute.
* Improves classification accuracy.
* Reduces tree complexity.

### Viva Questions

**Q:** Which attribute is selected first in a Decision Tree?
**A:** The attribute with the highest Information Gain.

**Q:** Which attribute had the highest IG in your experiment?
**A:** `checking_status`.

---

# 7. Why do we use Label Encoding? (5 Marks)

### Definition

**Label Encoding** converts categorical (text) values into numerical values so that machine learning algorithms can process the data.

### Why is it needed?

Most machine learning algorithms work only with numbers and cannot directly understand text values.

### Example

Before Encoding

| Gender |
| ------ |
| Male   |
| Female |

After Encoding

| Gender |
| ------ |
| 1      |
| 0      |

Another Example

| Credit Risk |
| ----------- |
| Good        |
| Bad         |

↓

| Credit Risk |
| ----------- |
| 1           |
| 0           |

### Advantages

* Converts text into numbers.
* Makes the dataset suitable for ML algorithms.
* Easy and fast preprocessing technique.

### Limitations

* Encoded numbers do **not** represent any order or importance.

### Viva Questions

**Q:** Why is Label Encoding used?
**A:** To convert categorical values into numeric values.

**Q:** Which Python class is used?
**A:** `LabelEncoder()`.

---

# 8. What is Cross Validation? (5 Marks)

### Definition

**Cross Validation** is a technique used to evaluate the performance of a machine learning model by dividing the dataset into multiple parts (folds).

### Working

* Divide the dataset into **k folds**.
* Train the model on **k−1 folds**.
* Test it on the remaining fold.
* Repeat until every fold has been used as the test set.
* Calculate the average accuracy.

### Example (5-Fold)

* Fold 1 → Test, remaining → Train
* Fold 2 → Test, remaining → Train
* Fold 3 → Test, remaining → Train
* Fold 4 → Test, remaining → Train
* Fold 5 → Test, remaining → Train

Average accuracy = Final Result

### Advantages

* Uses all data for training and testing.
* Produces reliable accuracy.
* Reduces bias due to one train-test split.

### Disadvantages

* Takes more computation time.
* Slightly slower than a single train-test split.

### Viva Questions

**Q:** What does `cv=5` mean?
**A:** 5-Fold Cross Validation.

**Q:** Why is Cross Validation used?
**A:** To obtain a more reliable estimate of model performance.

---

# 9. What is Overfitting? (5 Marks)

### Definition

**Overfitting** occurs when a model learns the training data too closely, including noise and small details, resulting in poor performance on new or unseen data.

### Characteristics

* Very high training accuracy.
* Low testing accuracy.
* Poor generalization.
* Common in very complex Decision Trees.

### Causes

* Large and deep Decision Trees.
* Small datasets.
* Too many unnecessary features.

### How to Reduce Overfitting?

* Pruning.
* Cross Validation.
* Feature Selection.
* Simpler models.

### Example

A student memorizes only previous exam questions instead of understanding the concepts. They score well on those questions but poorly on new questions. This is similar to overfitting.

### Viva Questions

**Q:** What is Overfitting?
**A:** Learning the training data too closely, resulting in poor performance on unseen data.

**Q:** How can Overfitting be reduced?
**A:** Pruning and Cross Validation.

---

# 10. What is Pruning? (5 Marks)

### Definition

**Pruning** is the process of removing unnecessary branches from a Decision Tree to make it simpler and improve its performance on new data.

### Why is Pruning Used?

* Reduces overfitting.
* Simplifies the Decision Tree.
* Improves generalization.
* Makes the model easier to understand.

### Types of Pruning

1. **Pre-Pruning**

   * Stops tree growth early.
   * Example: Limit `max_depth`.

2. **Post-Pruning (Reduced Error Pruning)**

   * Builds the full tree first.
   * Removes branches that do not improve accuracy.

### Advantages

* Smaller tree.
* Less overfitting.
* Better performance on unseen data.

### Disadvantages

* Excessive pruning can reduce accuracy (underfitting).

### Example

A Decision Tree with many unnecessary branches is simplified by removing branches that do not contribute to correct predictions.

### Viva Questions

**Q:** What is Pruning?
**A:** Removing unnecessary branches from a Decision Tree.

**Q:** Why is Pruning used?
**A:** To reduce overfitting and simplify the model.

**Q:** What is Reduced Error Pruning?
**A:** A post-pruning technique that removes branches if doing so does not reduce the model's accuracy.

---

## ⭐ Most Important Points to Remember

* **Information Gain** → Chooses the best attribute for splitting.
* **Label Encoding** → Converts categorical data into numeric values.
* **Cross Validation** → Evaluates the model using multiple folds.
* **Overfitting** → Model performs well on training data but poorly on new data.
* **Pruning** → Removes unnecessary branches to reduce overfitting and simplify the Decision Tree.

---

# 11. What is OneR? (5 Marks)

### Definition

**OneR (One Rule)** is a simple supervised classification algorithm that creates **one rule using only one attribute**. It selects the attribute that gives the **minimum classification error**.

### How OneR Works

1. Checks every attribute.
2. Calculates the error for each attribute.
3. Selects the attribute with the lowest error.
4. Creates one rule using that attribute.

### Example

Suppose the dataset has:

* Checking Status
* Age
* Duration
* Credit Amount

If **Checking Status** gives the lowest error, OneR uses only that attribute.

Example Rule:

```
IF checking_status = good
THEN class = Good Credit

ELSE
THEN class = Bad Credit
```

### Advantages

* Very simple.
* Easy to understand.
* Fast execution.
* Good baseline classifier.

### Disadvantages

* Uses only one attribute.
* Lower accuracy than Decision Trees.
* Cannot capture complex relationships.

### Applications

* Simple classification problems.
* Comparing with other classifiers.
* Baseline model.

### Viva Questions

**Q:** Why is it called OneR?

**A:** Because it creates only one rule using one attribute.

---

**Q:** Which attribute was selected in your experiment?

**A:** `checking_status`.

---

**Q:** Is OneR supervised?

**A:** Yes.

---

# 12. What is PART? (5 Marks)

### Definition

**PART** is a **rule-based supervised classification algorithm**. It generates **IF–THEN rules** by building partial Decision Trees.

### How PART Works

1. Builds a partial Decision Tree.
2. Converts one branch into an IF–THEN rule.
3. Removes covered instances.
4. Repeats until all data is covered.

### Example

```
IF checking_status = no checking
AND duration <= 24
THEN class = Good

IF credit_amount > 4000
THEN class = Bad
```

### Advantages

* Easy to understand.
* Produces readable rules.
* Usually performs almost as well as Decision Trees.

### Disadvantages

* May generate many rules.
* Slightly slower than OneR.

### Applications

* Credit approval.
* Medical diagnosis.
* Customer classification.

### Viva Questions

**Q:** What does PART generate?

**A:** IF–THEN Rules.

---

**Q:** Is PART supervised?

**A:** Yes.

---

**Q:** Difference between PART and J48?

**A:** J48 produces a Decision Tree, while PART produces IF–THEN rules.

---

# 13. What is K-Means? (5 Marks)

### Definition

**K-Means** is an **unsupervised learning algorithm** used to group similar data into **K clusters**.

### How K-Means Works

1. Choose the number of clusters (K).
2. Select K initial centroids.
3. Assign each data point to the nearest centroid.
4. Recalculate the centroids.
5. Repeat until the centroids no longer change.

### Example

Suppose there are 100 customers.

K = 2

Cluster 1 → Young Customers

Cluster 2 → Senior Customers

### Advantages

* Simple algorithm.
* Fast for large datasets.
* Easy to implement.

### Disadvantages

* Must choose K before running.
* Sensitive to initial centroid selection.
* Sensitive to outliers.

### Applications

* Customer segmentation.
* Market analysis.
* Image segmentation.
* Document clustering.

### Viva Questions

**Q:** Is K-Means supervised?

**A:** No, it is unsupervised.

---

**Q:** What is K?

**A:** Number of clusters.

---

**Q:** What is a centroid?

**A:** The center point of a cluster.

---

# 14. What is SVM? (5 Marks)

### Definition

**SVM (Support Vector Machine)** is a **supervised learning algorithm** used for classification and regression. It finds the **best hyperplane** that separates different classes.

### How SVM Works

1. Takes labeled training data.
2. Finds the optimal hyperplane.
3. Maximizes the margin between classes.
4. Classifies new data using the hyperplane.

### Important Terms

**Hyperplane:** Decision boundary separating classes.

**Support Vectors:** Data points closest to the hyperplane that determine its position.

### Advantages

* High accuracy.
* Works well with high-dimensional data.
* Effective for complex classification problems.

### Disadvantages

* Slower for very large datasets.
* Harder to interpret than Decision Trees.
* Choice of kernel affects performance.

### Applications

* Face recognition.
* Spam detection.
* Disease diagnosis.
* Credit risk prediction.

### Viva Questions

**Q:** What is a hyperplane?

**A:** A decision boundary that separates different classes.

---

**Q:** What are support vectors?

**A:** Data points closest to the hyperplane.

---

**Q:** Which WEKA algorithm implements SVM?

**A:** SMO.

---

# 15. Difference between J48, PART, and OneR (5 Marks)

| Feature         | J48           | PART           | OneR         |
| --------------- | ------------- | -------------- | ------------ |
| Algorithm Type  | Decision Tree | Rule-Based     | Rule-Based   |
| Output          | Tree          | IF–THEN Rules  | One Rule     |
| Attributes Used | Multiple      | Multiple       | One          |
| Accuracy        | High          | Medium to High | Lower        |
| Complexity      | Moderate      | Moderate       | Very Simple  |
| WEKA Classifier | trees → J48   | rules → PART   | rules → OneR |

### Explanation

### J48

* WEKA implementation of the C4.5 Decision Tree.
* Produces a tree structure.
* Usually provides the highest accuracy.

### PART

* Produces IF–THEN rules instead of a tree.
* Rules are easy to understand.
* Accuracy is usually close to J48.

### OneR

* Uses only one attribute.
* Simplest classifier.
* Easy to understand.
* Usually has lower accuracy.

### Ranking

1. **J48** – Highest accuracy.
2. **PART** – Slightly lower or similar accuracy.
3. **OneR** – Lowest accuracy but simplest.

### Viva Questions

**Q:** Which classifier produces a Decision Tree?

**A:** J48.

---

**Q:** Which classifier produces IF–THEN rules?

**A:** PART.

---

**Q:** Which classifier uses only one attribute?

**A:** OneR.

---

**Q:** Which classifier generally has the highest accuracy?

**A:** J48.

---

## ⭐ Final 20 Most Expected Viva Questions

1. What is Data Mining?
2. What is KDD?
3. Difference between KDD and Data Mining.
4. Difference between Classification and Clustering.
5. Difference between Supervised and Unsupervised Learning.
6. What is a Decision Tree?
7. What is Information Gain?
8. Why do we use Label Encoding?
9. What is Cross Validation?
10. What is Overfitting?
11. What is Pruning?
12. What is OneR?
13. What is PART?
14. Difference between PART and OneR.
15. What is K-Means?
16. What is a Cluster?
17. What is a Centroid?
18. What is SVM?
19. What is a Hyperplane?
20. Difference between **J48, PART, and OneR**.
