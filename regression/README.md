Note: Only multilinear.cpp implements feature scaling, the other programs may crash under different datasets.

Regression project: https://github.com/longwatermelon/graph-read

<details>
<summary><code>src/linear.cpp</code></summary>
Fits a linear function through a dataset.<br>
<img src=https://user-images.githubusercontent.com/73869536/210283143-b2324de7-6153-4727-9663-9024567d1a69.png>
</details>

<details>
<summary><code>src/multilinear.cpp</code></summary>
Linear regression but with multiple features.<br>
<pre><code>Iteration 100: w = [107.730453, -19.863569, -31.097254, -38.512470], b = 363.145355
Iteration 200: w = [110.466301, -21.219683, -32.654575, -37.988583], b = 363.155975
Iteration 300: w = [110.557259, -21.265566, -32.705429, -37.970764], b = 363.155975
Iteration 400: w = [110.560287, -21.267094, -32.707123, -37.970188], b = 363.155975
Iteration 500: w = [110.560303, -21.267113, -32.707123, -37.970181], b = 363.155975
Price prediction of house with size = 1200, bedrooms = 3, floors = 1, age = 40: $318709.00
</code></pre>
</details>

<details>
<summary><code>src/logistic.cpp</code></summary>
Fits a sigmoid function through a dataset. (Binary classification)<br>
<img src=https://user-images.githubusercontent.com/73869536/210283442-1735aef4-21eb-4952-9fcb-d433c14290dc.png>
</details>

<details>
<summary><code>src/multilogistic.cpp</code></summary>
Binary classification with two features, cutoff is at <code>f_wb(x) = 0.5</code><br>
<img src=https://user-images.githubusercontent.com/73869536/210283509-32fae22e-4195-47c2-b3d6-3aa69e311e79.png>
</details>

<details>
<summary><code>src/softmax.cpp</code></summary>
Multiclass classification with two features.<br>
<img src=https://user-images.githubusercontent.com/73869536/210283610-a8d8087c-f490-4378-a707-d00734a29e91.png>
</details>
