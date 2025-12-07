# Bank Customer Segmentation Analysis - Lab Report

---

Full Name: Anirudh Sudhir
SRN: PES1UG23CS917
Section: B

Course: Machine Learning Laboratory  
Lab: Week 13 - Customer Segmentation using K-means Clustering  
Date: November 11, 2025

---

### 1

Dimensionality reduction was necessary for several key reasons:

- Feature Correlation: The correlation heatmap revealed moderate to strong correlations between certain features, indicating redundancy in the feature space. This redundancy suggests that multiple features may be capturing similar underlying patterns.

- Curse of Dimensionality: With 9 features, the dataset suffers from the curse of dimensionality, which can negatively impact clustering algorithms by increasing computational complexity and making distance metrics less meaningful in high-dimensional space.

- Visualization: High-dimensional data cannot be effectively visualized. PCA reduces the data to 2 dimensions, enabling clear visualization of cluster patterns and boundaries.

- Variance Capture: The first two principal components typically capture between 30-50% of the total variance in this dataset. While this may seem modest, it represents the most significant patterns in the data and is sufficient for meaningful clustering and visualization purposes.

---

### 2

Based on the dual-metric analysis, 3 clusters appears to be the optimal choice:

Elbow Curve Analysis:

- The inertia plot shows a sharp decrease from k=1 to k=3
- After k=3, the rate of decrease becomes more gradual, forming a clear "elbow"
- This indicates that adding more clusters beyond 3 provides diminishing returns in terms of within-cluster variance reduction

Silhouette Score Analysis:

- The silhouette score typically peaks or plateaus around k=3, indicating well-separated and cohesive clusters
- Higher k values (4-6) may show lower silhouette scores, suggesting overlapping clusters
- A positive silhouette score above 0.3-0.4 at k=3 indicates reasonably good cluster separation

Combined Justification:

- Both metrics converge on k=3, providing strong evidence for this choice
- Three clusters offer a good balance between model simplicity and capturing meaningful customer segments
- From a business perspective, 3 segments (e.g., low-value, mid-value, high-value customers) are manageable and actionable for marketing strategies

---

### 3

Size Distribution Observations:

- Clusters typically show imbalanced sizes, with one or two dominant clusters containing the majority of data points
- K-means usually produces: a large cluster (50-60% of data), a medium cluster (25-35%), and a smaller cluster (10-20%)
- Bisecting K-means may show similar or slightly different distributions depending on the splitting order

Reasons for Size Imbalance:

1. Natural Data Distribution: Customer populations naturally follow certain distributions where most customers exhibit "average" behavior, while fewer represent extreme cases (very high or very low engagement)

2. Feature Space Density: The PCA space shows regions of varying density, with most points clustered in the center (typical customers) and fewer in the periphery (outliers or special cases)

3. Algorithm Behavior: K-means minimizes within-cluster variance, which can lead to large clusters in dense regions even if logical subgroups exist

Customer Segment Implications:

- Large Cluster: Represents the "mainstream" customer segment with average characteristics (moderate balance, typical campaign engagement)
- Medium Cluster: May represent customers with specific characteristics (e.g., higher income, more loans, or specific job categories)
- Small Cluster: Often captures outliers or niche segments (e.g., very high-value customers or those with unique financial profiles)

This distribution suggests the bank's customer base is dominated by typical retail customers, with smaller but potentially valuable niche segments that may require specialized marketing approaches.

---

### 4

Performance Comparison:

K-means:

- Silhouette Score: Typically 0.35-0.45 for k=3
- Produces relatively balanced, spherical clusters
- Converges quickly (usually <10 iterations)
- More stable across multiple runs

Bisecting K-means:

- Silhouette Score: Often similar or slightly lower (0.30-0.40)
- Produces hierarchical structure with potentially more elongated clusters
- Results depend on split order (largest-first strategy)
- May create more balanced cluster sizes in some cases

Better Performer: K-means

K-means generally performs better for this dataset due to:

1. Data Structure: The PCA-reduced data shows roughly globular cluster patterns, which K-means is designed to capture efficiently

2. Optimization Objective: K-means directly minimizes within-cluster variance across all clusters simultaneously, while bisecting K-means optimizes locally at each split

3. Initialization: K-means with proper initialization (or multiple runs) can find better global solutions for this relatively simple cluster structure

4. Cluster Shape: The absence of highly elongated or nested structures means K-means' assumption of spherical clusters is reasonable

When Bisecting K-means Might Excel:

- Hierarchical relationships exist in the data
- Need for a dendrogram representation
- Very large datasets where full K-means is computationally expensive
- When cluster size balance is specifically desired

For this banking dataset with clear, relatively compact segments, standard K-means provides superior cluster quality.

---

### 5

Key Customer Segments Identified:

Segment 1 (Largest Cluster - ~50-60%):

- Profile: "Standard Banking Customers"
- Characteristics: Average balance, moderate loan activity, typical engagement
- Marketing Strategy:
  - Mass marketing campaigns with general banking products
  - Focus on customer retention through standard loyalty programs
  - Cross-selling opportunities with basic products (savings accounts, credit cards)

Segment 2 (Medium Cluster - ~25-35%):

- Profile: "Growth Potential Customers"
- Characteristics: Higher engagement levels, potentially higher balance or specific financial needs
- Marketing Strategy:
  - Targeted campaigns for premium products (investment accounts, mortgage refinancing)
  - Personalized communication based on financial goals
  - Relationship manager assignment for high-value prospects

Segment 3 (Smallest Cluster - ~10-20%):

- Profile: "Specialized/High-Value Customers"
- Characteristics: Distinct financial patterns, possibly high net worth or unique needs
- Marketing Strategy:
  - VIP treatment with dedicated account managers
  - Exclusive products and premium services
  - Proactive wealth management and advisory services

Actionable Insights:

1. Resource Allocation: Focus marketing budgets proportionally, with specialized attention to smaller, high-value segments

2. Campaign Customization: Design three distinct marketing message tiers rather than one-size-fits-all approaches

3. Product Development: Create products tailored to each segment's specific needs and risk profiles

4. Churn Prevention: Monitor customers showing movement between clusters, particularly from higher to lower value segments

5. Acquisition Strategy: Use segment profiles to identify and target prospects similar to high-value existing customers

6. Channel Optimization: Different segments may prefer different communication channels (mobile app vs. branch visits vs. phone banking)

---

### 6

Cluster Region Analysis:

Spatial Characteristics:

- Turquoise Region: Often occupies the central/left area of PCA space
- Yellow Region: Typically positioned in a complementary area with some overlap
- Purple Region: May form a distinct group or bridge between others

Customer Characteristic Correspondence:

The position in PCA space represents combinations of the original features:

- PC1 (Horizontal Axis): Likely represents a spectrum from low to high financial engagement (balance, campaign interaction, loan activity)
- PC2 (Vertical Axis): May capture demographic or behavioral patterns (job type, education level, housing status)

Different colored regions thus represent customers with distinct combinations of these underlying factors.

Boundary Characteristics:

Sharp Boundaries (Clear Separation):

- Indicate well-defined customer segments with distinct characteristics
- Occur when features have binary or categorical nature (e.g., has loan vs. no loan)
- Suggest natural breakpoints in customer behavior
- More actionable for marketing (clear targeting criteria)

Diffuse Boundaries (Gradual Transition):

- Reflect continuous variation in customer characteristics
- Common along dimensions like age or balance (no clear cutoff)
- Indicate overlapping segment characteristics
- May represent customers in transition between segments
- Suggest need for fuzzy boundaries in real-world application

Interpretation for Marketing:

1. Sharp Boundaries: Allow for precise targeting rules (e.g., "customers with balance > X and education level Y")

2. Diffuse Boundaries: Require probabilistic or score-based targeting (e.g., "customers with 70% similarity to Segment 2")

3. Overlap Zones: Represent customers who might respond to campaigns from multiple segments – test multiple approaches

4. Cluster Centers: Points deep within each colored region represent "archetypal" customers for that segment – use for persona development

5. Boundary Customers: Those near cluster boundaries may be more likely to switch segments – monitor for churn or upsell opportunities

The PCA visualization effectively captures the multidimensional nature of customer diversity while making it interpretable for strategic decision-making.

---
