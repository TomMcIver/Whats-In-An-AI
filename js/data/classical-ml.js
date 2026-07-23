/* Classical machine learning — the models that predate deep learning
   and still win on most tabular problems. */

(function (NS) {
  'use strict';
  NS.categories = NS.categories || [];

  NS.categories.push({
    id: 'classical-ml',
    name: 'Classical ML',
    order: 1,
    models: [

    {
      id: 'linear-regression',
      name: 'Linear Regression',
      year: '1805',
      learns: 'Supervised',
      dataFor: 'Tabular, numeric target',
      tagline: 'Draw the straight line that sits closest to all the points.',
      diagram: { type: 'scatter', boundary: 'line', b: [], caption: 'one line, fitted to minimise squared error' },
      how: [
        'Assume the answer is a *weighted sum* of the inputs: price = w₁·size + w₂·rooms + b.',
        'Measure how wrong the line is using *squared error* — the vertical gap to each point, squared.',
        'Adjust the weights to make that total error as small as possible. For a straight line there is an exact formula, so no iteration is needed.',
        'The learned weights are directly readable: w₁ is literally "pounds per square metre".'
      ],
      keyIdea: 'The simplest possible model — and the baseline every other model has to beat. If a straight line already explains your data, nothing more complicated is justified.',
      strength: 'Interpretability. You can explain exactly why it predicted what it did.',
      limitation: 'Anything genuinely curved. It can only ever draw a flat surface.',
      usedFor: ['forecasting', 'pricing', 'risk scoring', 'baselines']
    },

    {
      id: 'logistic-regression',
      name: 'Logistic Regression',
      year: '1958',
      learns: 'Supervised',
      dataFor: 'Tabular, categorical target',
      tagline: 'Linear regression bent into a probability between 0 and 1.',
      diagram: { type: 'scatter', boundary: 'line', caption: 'a line, but the output is "how confident", not "how much"' },
      how: [
        'Compute a weighted sum of the inputs, exactly as linear regression does.',
        'Squash that number through the *sigmoid* function, which maps anything to the range 0–1.',
        'Read the result as a probability: 0.83 means "83% likely to be class A".',
        'Train by adjusting weights to maximise the probability assigned to the correct answers.'
      ],
      keyIdea: 'Despite the name it is a *classifier*. The sigmoid is the only thing separating it from linear regression, and it is what turns a magnitude into a confidence.',
      strength: 'Calibrated probabilities, not just labels — vital in medicine and credit.',
      limitation: 'Still draws a straight boundary. Classes that interleave will defeat it.',
      usedFor: ['credit scoring', 'medical diagnosis', 'churn prediction', 'A/B testing']
    },

    {
      id: 'decision-tree',
      name: 'Decision Tree',
      year: '1963',
      learns: 'Supervised',
      dataFor: 'Tabular, mixed types',
      tagline: 'A flowchart of yes/no questions, learned from the data.',
      diagram: { type: 'tree', depth: 4, tests: ['age < 30?', 'income > 40k?', 'owns home?'] },
      how: [
        'Look at every feature and every possible threshold, and ask which single split best separates the classes.',
        'Measure "best" with *Gini impurity* or *entropy* — both reward splits that produce purer groups.',
        'Take the winning split, then repeat the whole process independently on each side.',
        'Stop when a group is pure, too small, or the tree hits its depth limit. Those end points are the leaves that hold predictions.'
      ],
      keyIdea: 'It carves the feature space into rectangles. That makes it readable by a human — you can print the tree and follow it by hand — but also brittle.',
      strength: 'Handles mixed numeric and categorical data with no scaling or preprocessing.',
      limitation: 'Wildly unstable. Change a few rows and you can get a completely different tree.',
      usedFor: ['rule extraction', 'triage systems', 'feature selection', 'explainable decisions']
    },

    {
      id: 'random-forest',
      name: 'Random Forest',
      year: '2001',
      learns: 'Supervised (ensemble)',
      dataFor: 'Tabular, mixed types',
      tagline: 'Hundreds of deliberately mediocre trees, averaged into one strong answer.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Data', kind: 'in' },
          { label: 'Tree 1', kind: 'hidden' },
          { label: 'Tree 2', kind: 'hidden' },
          { label: 'Tree N', kind: 'hidden' },
          { label: 'Vote', kind: 'out' }
        ],
        caption: 'each tree sees a different random slice of rows and columns'
      },
      how: [
        'Build hundreds of decision trees, but *handicap each one*: give it a random sample of the rows (bagging).',
        'At every split, let it consider only a random subset of the features, so trees cannot all seize on the same dominant signal.',
        'Each tree ends up overfitted and individually unreliable — but wrong in its *own* direction.',
        'Average their predictions (or take a majority vote). The independent errors cancel; the shared signal survives.'
      ],
      keyIdea: 'Variance cancels when errors are uncorrelated. The randomness is not a compromise — it is the entire mechanism, deliberately making each tree worse so the ensemble is better.',
      strength: 'Excellent accuracy with almost no tuning. A superb default for tabular data.',
      limitation: 'You lose the readability of a single tree, and it is slow to serve at scale.',
      usedFor: ['tabular prediction', 'feature importance', 'fraud detection', 'bioinformatics']
    },

    {
      id: 'gradient-boosting',
      name: 'Gradient Boosting',
      aka: 'XGBoost, LightGBM, CatBoost',
      year: '1999',
      learns: 'Supervised (ensemble)',
      dataFor: 'Tabular, mixed types',
      tagline: 'Trees built in sequence, each one fixing the last one\'s mistakes.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Guess', kind: 'in' },
          { label: 'Tree 1\non errors', kind: 'hidden' },
          { label: 'Tree 2\non errors', kind: 'hidden' },
          { label: 'Sum', kind: 'out' }
        ],
        caption: 'each tree is trained on what remains wrong'
      },
      how: [
        'Start with a trivial prediction — usually just the average of the target.',
        'Compute the *residuals*: how far off that prediction is for every row.',
        'Train a small tree to predict those residuals, and add a fraction of its output (the learning rate) to the running prediction.',
        'Recompute residuals and repeat, often for hundreds of rounds. Each tree only has to explain what is left over.'
      ],
      keyIdea: 'Where a random forest builds trees *in parallel and averages*, boosting builds them *in sequence and adds*. That sequential correction is why it usually edges out the forest — and why it overfits if left unchecked.',
      strength: 'State of the art on tabular data. Wins most Kaggle competitions that are not images or text.',
      limitation: 'Sensitive to hyperparameters, and will happily memorise noise if you let it run too long.',
      usedFor: ['ranking', 'click prediction', 'credit risk', 'competitions']
    },

    {
      id: 'svm',
      name: 'Support Vector Machine',
      aka: 'SVM',
      year: '1992',
      learns: 'Supervised',
      dataFor: 'Tabular, high-dimensional',
      tagline: 'Find the boundary with the widest possible gap either side.',
      diagram: { type: 'scatter', boundary: 'margin', caption: 'only the points nearest the boundary matter' },
      how: [
        'Among all the lines that separate the two classes, look for the one with the *largest margin* — the biggest empty corridor around it.',
        'Only the handful of points sitting on the edge of that corridor affect the answer. These are the *support vectors*; every other point could be deleted without changing the result.',
        'If the classes cannot be separated by a straight line, apply the *kernel trick*: implicitly project into a higher-dimensional space where they can be.',
        'The kernel computes distances in that space without ever building it, which is what makes the trick affordable.'
      ],
      keyIdea: 'Maximising the margin is a bet about generalisation — a boundary with room to spare is more likely to survive contact with unseen data.',
      strength: 'Very strong when you have many features and few samples, such as gene expression data.',
      limitation: 'Scales badly past ~100k rows, and gives no probability estimate without extra work.',
      usedFor: ['text classification', 'image classification (pre-2012)', 'bioinformatics']
    },

    {
      id: 'knn',
      name: 'k-Nearest Neighbours',
      aka: 'k-NN',
      year: '1951',
      learns: 'Supervised (instance-based)',
      dataFor: 'Tabular, numeric',
      tagline: 'To classify something new, look at what its closest neighbours are.',
      diagram: { type: 'scatter', boundary: 'step', caption: 'the boundary is implied by the data, never computed' },
      how: [
        'There is no training step at all. The model simply stores every training example.',
        'To predict, measure the distance from the new point to every stored point.',
        'Take the *k* closest ones — five, say — and let them vote on the label.',
        'A small k gives a jagged, noise-sensitive boundary; a large k smooths it out, eventually to the point of ignoring local structure.'
      ],
      keyIdea: 'A *lazy* learner: it does no work until asked a question, then does all of it at once. All the cost is moved from training to prediction.',
      strength: 'Zero training time, and the decision boundary can be arbitrarily complex.',
      limitation: 'Prediction is slow and memory-hungry, and it collapses in high dimensions where everything is roughly equidistant.',
      usedFor: ['recommendation', 'anomaly detection', 'imputation', 'quick baselines']
    },

    {
      id: 'naive-bayes',
      name: 'Naive Bayes',
      year: '1960s',
      learns: 'Supervised (probabilistic)',
      dataFor: 'Counts, text',
      tagline: 'Count how often each clue appears in each class, then apply Bayes\' rule.',
      diagram: {
        type: 'blocks',
        items: [
          { label: 'Features', kind: 'in' },
          { label: 'P(f | class)\nper feature', kind: 'hidden' },
          { label: 'Multiply\n× prior', kind: 'hidden' },
          { label: 'Class', kind: 'out' }
        ],
        caption: 'assumes every feature is independent of the others'
      },
      how: [
        'For each class, count how often each feature appears — how often "free" shows up in spam versus normal mail.',
        'Assume, quite wrongly, that every feature is *independent* given the class. This is the "naive" part.',
        'That assumption lets you simply multiply the individual probabilities instead of modelling how features interact.',
        'Multiply by the prior (how common the class is overall) and pick the highest score.'
      ],
      keyIdea: 'The independence assumption is essentially always false — in real text, "New" and "York" are deeply dependent. It works anyway, because for *ranking* classes the errors tend to affect all classes similarly.',
      strength: 'Extremely fast, needs little data, and is a genuinely strong text baseline.',
      limitation: 'Its probability estimates are badly calibrated — treat them as scores, not real probabilities.',
      usedFor: ['spam filtering', 'sentiment analysis', 'document sorting']
    },

    {
      id: 'kmeans',
      name: 'k-Means Clustering',
      year: '1957',
      learns: 'Unsupervised',
      dataFor: 'Tabular, numeric',
      tagline: 'Guess k centres, assign every point to the nearest, move the centres, repeat.',
      diagram: {
        type: 'scatter', centroids: [[.24, .36], [.76, .68]],
        caption: 'centres drift until they stop moving'
      },
      how: [
        'Pick *k* — the number of clusters you want — and scatter k centre points at random.',
        'Assign every data point to whichever centre is nearest.',
        'Move each centre to the average position of the points assigned to it.',
        'Repeat steps 2 and 3. Points get reassigned, centres shift, and within a few dozen rounds nothing moves any more.'
      ],
      keyIdea: 'You must choose k in advance, which is the awkward part — the algorithm cannot tell you how many groups exist, only how to place k of them.',
      strength: 'Fast, simple, and scales to very large datasets.',
      limitation: 'Assumes round, equally-sized clusters. Elongated or nested shapes break it completely.',
      usedFor: ['customer segmentation', 'image compression', 'document grouping']
    },

    {
      id: 'dbscan',
      name: 'DBSCAN',
      year: '1996',
      learns: 'Unsupervised',
      dataFor: 'Tabular, spatial',
      tagline: 'Clusters are dense regions; anything in a sparse gap is noise.',
      diagram: {
        type: 'scatter',
        a: [[.14,.28],[.20,.20],[.26,.32],[.18,.42],[.28,.24],[.22,.38]],
        b: [[.72,.66],[.80,.58],[.68,.76],[.84,.70],[.76,.62],[.66,.68]],
        c: [[.50,.14],[.92,.22]],
        caption: 'the two isolated points are labelled noise, not forced into a cluster'
      },
      how: [
        'Pick a radius *ε* and a minimum neighbour count. A point with enough neighbours inside that radius is a *core point*.',
        'Core points that are within ε of each other are chained into the same cluster.',
        'Points near a core point but without enough neighbours of their own join as *border points*.',
        'Anything left over is labelled *noise* — and stays that way, rather than being forced into a cluster.'
      ],
      keyIdea: 'Unlike k-means you never state how many clusters exist; density decides. It is also the rare clustering method that is allowed to say "this point belongs to nothing".',
      strength: 'Finds arbitrarily shaped clusters and identifies outliers as a by-product.',
      limitation: 'Struggles when clusters have very different densities, and ε is fiddly to choose.',
      usedFor: ['spatial data', 'anomaly detection', 'GPS trace clustering']
    },

    {
      id: 'hierarchical-clustering',
      name: 'Hierarchical Clustering',
      year: '1963',
      learns: 'Unsupervised',
      dataFor: 'Tabular, any distance metric',
      tagline: 'Repeatedly merge the two closest groups until everything is one tree.',
      diagram: { type: 'tree', depth: 4, caption: 'read the tree at any height to get that many clusters' },
      how: [
        'Start with every point as its own cluster of one.',
        'Find the two closest clusters and merge them.',
        'Repeat until a single cluster contains everything, recording the merge order.',
        'The result is a *dendrogram* — a tree you can cut at any height to get however many clusters you want.'
      ],
      keyIdea: 'You do not commit to a cluster count up front. The full hierarchy is computed once, and you choose the granularity afterwards by deciding where to cut.',
      strength: 'Produces a complete, inspectable structure rather than a flat assignment.',
      limitation: 'Cost grows roughly with the cube of the sample count — impractical beyond tens of thousands of rows.',
      usedFor: ['taxonomy building', 'gene expression', 'document hierarchies']
    },

    {
      id: 'gmm',
      name: 'Gaussian Mixture Model',
      aka: 'GMM',
      year: '1977',
      learns: 'Unsupervised (probabilistic)',
      dataFor: 'Tabular, numeric',
      tagline: 'Assume the data is several overlapping bell curves, and work out their shapes.',
      diagram: {
        type: 'scatter', centroids: [[.26, .34], [.72, .66]],
        caption: 'each point gets a probability of belonging to each cluster'
      },
      how: [
        'Assume the data was produced by *k* Gaussian distributions blended together, each with its own centre, spread and orientation.',
        'Guess those parameters, then compute how likely each point is to have come from each Gaussian (the E-step).',
        'Update each Gaussian using all points, weighted by those likelihoods (the M-step).',
        'Alternate the two steps — *expectation-maximisation* — until the fit stops improving.'
      ],
      keyIdea: 'A soft version of k-means. A point is not in cluster 2; it is 70% cluster 2 and 30% cluster 3. Because each Gaussian has its own covariance, clusters can be stretched and tilted rather than round.',
      strength: 'Handles elliptical, overlapping clusters and gives genuine membership probabilities.',
      limitation: 'Needs k up front, and can converge to a poor local optimum depending on initialisation.',
      usedFor: ['speaker identification', 'density estimation', 'soft segmentation']
    },

    {
      id: 'pca',
      name: 'Principal Component Analysis',
      aka: 'PCA',
      year: '1901',
      learns: 'Unsupervised (dimensionality reduction)',
      dataFor: 'Tabular, numeric',
      tagline: 'Rotate the data so the first axis captures the most variation.',
      diagram: {
        type: 'hourglass',
        labels: ['1000 features', 'rotate', '50 components', 'rebuild', '1000 features'],
        heights: [120, 80, 34, 80, 120],
        caption: 'keep the directions that vary most, discard the rest'
      },
      how: [
        'Centre the data, then find the direction along which it varies most. That is the first principal component.',
        'Find the next direction that captures the most *remaining* variance, subject to being perpendicular to the first.',
        'Continue until you have as many components as original features, ordered by how much variance each explains.',
        'Keep only the top handful. If 50 components explain 95% of the variance, you have cut 1000 features to 50 for a 5% loss.'
      ],
      keyIdea: 'It is a rotation, not a selection. The new axes are blends of the originals — which is why they compress well but are hard to interpret.',
      strength: 'Fast, deterministic, and reverses cleanly so you can reconstruct approximations.',
      limitation: 'Only captures *linear* structure. A spiral will defeat it entirely.',
      usedFor: ['compression', 'noise reduction', 'visualisation', 'preprocessing']
    },

    {
      id: 'tsne-umap',
      name: 't-SNE & UMAP',
      year: '2008 / 2018',
      learns: 'Unsupervised (visualisation)',
      dataFor: 'High-dimensional numeric',
      tagline: 'Squash high-dimensional data to 2D while keeping neighbours together.',
      diagram: {
        type: 'scatter',
        a: [[.12,.22],[.18,.16],[.22,.30],[.14,.36],[.26,.20],[.20,.42]],
        b: [[.74,.68],[.82,.60],[.70,.78],[.86,.72],[.78,.64],[.68,.74]],
        c: [[.46,.32],[.54,.26],[.50,.40],[.58,.36]],
        caption: 'distances within a blob are meaningful; distances between them are not'
      },
      how: [
        'Measure which points are neighbours in the original high-dimensional space.',
        'Scatter the points randomly on a 2D canvas.',
        'Nudge them so that points which were neighbours end up close, and points which were not end up apart.',
        'Repeat until the 2D layout preserves the neighbourhood structure as well as it can.'
      ],
      keyIdea: 'These preserve *local* structure only. The size of a blob and the gap between two blobs are largely artefacts — reading them as meaningful is the most common mistake people make with these plots.',
      strength: 'Reveals cluster structure in data with hundreds of dimensions.',
      limitation: 'Non-deterministic, slow, and easy to over-interpret. Use for exploration, never as evidence.',
      usedFor: ['embedding visualisation', 'single-cell genomics', 'exploratory analysis']
    },

    {
      id: 'isolation-forest',
      name: 'Isolation Forest',
      year: '2008',
      learns: 'Unsupervised (anomaly detection)',
      dataFor: 'Tabular, numeric',
      tagline: 'Outliers are the points you can isolate with very few random cuts.',
      diagram: { type: 'tree', depth: 3, caption: 'anomalies fall out of the tree after only a few splits' },
      how: [
        'Pick a random feature and a random split value, cutting the data in two.',
        'Keep cutting recursively until every point sits alone in its own region.',
        'Record how many cuts each point needed. Points in dense areas need many; isolated points need few.',
        'Average the depth across many random trees. Consistently shallow points are the anomalies.'
      ],
      keyIdea: 'It inverts the usual approach. Instead of modelling what normal looks like and flagging deviations, it directly exploits the fact that anomalies are *easy to separate*.',
      strength: 'Linear time, no distance computations, works well in high dimensions.',
      limitation: 'Assumes anomalies are rare and distinct; struggles when they form their own dense cluster.',
      usedFor: ['fraud detection', 'intrusion detection', 'quality control']
    }

    ]
  });
})(window.WIAA = window.WIAA || {});
