#include "matrix.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <stdexcept>

// build a matrix from a brace literal, every row must be the same length
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init)
    : rows(sz(init)), cols(init.size() ? sz(*init.begin()) : 0),
      data(vec2<double>(rows, cols)) {
    int i = 0;
    for (const auto &row : init) {
        assert(sz(row) == cols && "all rows must have the same length");
        int j = 0;
        for (double v : row)
            data[i][j++] = v;
        i++;
    }
}

// ---- factory helpers ----

// matrix of all zeros
Matrix Matrix::zeros(int rows, int cols) {
    return Matrix(rows, cols);
}

// matrix of all ones
Matrix Matrix::ones(int rows, int cols) {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = 1.0;
    return m;
}

// n x n identity matrix
Matrix Matrix::identity(int n) {
    Matrix m(n, n);
    for (int i = 0; i < n; i++)
        m.data[i][i] = 1.0;
    return m;
}

// square matrix with the given values on the main diagonal
Matrix Matrix::from_diagonal(const vec<double> &d) {
    int n = sz(d);
    Matrix m(n, n);
    for (int i = 0; i < n; i++)
        m.data[i][i] = d[i];
    return m;
}

// ---- element access ----

// mutable access to entry (i, j)
double &Matrix::operator()(int i, int j) {
    assert(i >= 0 && i < rows && j >= 0 && j < cols);
    return data[i][j];
}

// read-only access to entry (i, j)
double Matrix::operator()(int i, int j) const {
    assert(i >= 0 && i < rows && j >= 0 && j < cols);
    return data[i][j];
}

// ---- shape helpers ----

// true if both matrices have identical dimensions
bool Matrix::same_shape(const Matrix &o) const {
    return rows == o.rows && cols == o.cols;
}

// true if the matrix is square
bool Matrix::is_square() const {
    return rows == cols;
}

// ---- element-wise arithmetic ----

// element-wise sum of two matrices
Matrix Matrix::operator+(const Matrix &o) const {
    assert(same_shape(o));
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] + o.data[i][j];
    return m;
}

// element-wise difference of two matrices
Matrix Matrix::operator-(const Matrix &o) const {
    assert(same_shape(o));
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] - o.data[i][j];
    return m;
}

// in-place element-wise addition
Matrix &Matrix::operator+=(const Matrix &o) {
    assert(same_shape(o));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] += o.data[i][j];
    return *this;
}

// in-place element-wise subtraction
Matrix &Matrix::operator-=(const Matrix &o) {
    assert(same_shape(o));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] -= o.data[i][j];
    return *this;
}

// element-wise (hadamard) product
Matrix Matrix::hadamard(const Matrix &o) const {
    assert(same_shape(o));
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] * o.data[i][j];
    return m;
}

// element-wise division
Matrix Matrix::ediv(const Matrix &o) const {
    assert(same_shape(o));
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] / o.data[i][j];
    return m;
}

// ---- unary negation ----

// negate every entry
Matrix Matrix::operator-() const {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = -data[i][j];
    return m;
}

// ---- scalar arithmetic ----

// add a scalar to every entry
Matrix Matrix::operator+(double s) const {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] + s;
    return m;
}

// subtract a scalar from every entry
Matrix Matrix::operator-(double s) const {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] - s;
    return m;
}

// multiply every entry by a scalar
Matrix Matrix::operator*(double s) const {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] * s;
    return m;
}

// divide every entry by a scalar
Matrix Matrix::operator/(double s) const {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = data[i][j] / s;
    return m;
}

// in-place scalar addition
Matrix &Matrix::operator+=(double s) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] += s;
    return *this;
}

// in-place scalar subtraction
Matrix &Matrix::operator-=(double s) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] -= s;
    return *this;
}

// in-place scalar multiplication
Matrix &Matrix::operator*=(double s) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] *= s;
    return *this;
}

// in-place scalar division
Matrix &Matrix::operator/=(double s) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] /= s;
    return *this;
}

// scalar on the left: s + each entry (commutative)
Matrix operator+(double s, const Matrix &m) {
    return m + s;
}

// scalar on the left: s - each entry
Matrix operator-(double s, const Matrix &m) {
    Matrix r(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            r.data[i][j] = s - m.data[i][j];
    return r;
}

// scalar on the left: s * each entry (commutative)
Matrix operator*(double s, const Matrix &m) {
    return m * s;
}

// scalar on the left: s / each entry
Matrix operator/(double s, const Matrix &m) {
    Matrix r(m.rows, m.cols);
    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            r.data[i][j] = s / m.data[i][j];
    return r;
}

// ---- matrix multiplication ----

// standard matrix product, dimensions must agree
Matrix Matrix::operator*(const Matrix &o) const {
    assert(cols == o.rows);
    Matrix m(rows, o.cols);
    for (int i = 0; i < rows; i++)
        for (int k = 0; k < cols; k++) {
            double a = data[i][k];
            for (int j = 0; j < o.cols; j++)
                m.data[i][j] += a * o.data[k][j];
        }
    return m;
}

// ---- comparison ----

// exact element-wise equality (same shape and values)
bool Matrix::operator==(const Matrix &o) const {
    if (!same_shape(o))
        return false;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (data[i][j] != o.data[i][j])
                return false;
    return true;
}

bool Matrix::operator!=(const Matrix &o) const {
    return !(*this == o);
}

// ---- structural operations ----

// transpose: swap rows and columns
Matrix Matrix::transpose() const {
    Matrix m(cols, rows);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[j][i] = data[i][j];
    return m;
}

// copy with one row and one column removed (used for minors)
Matrix Matrix::submatrix(int skip_row, int skip_col) const {
    assert(skip_row >= 0 && skip_row < rows && skip_col >= 0 && skip_col < cols);
    Matrix m(rows - 1, cols - 1);
    for (int i = 0, di = 0; i < rows; i++) {
        if (i == skip_row)
            continue;
        for (int j = 0, dj = 0; j < cols; j++) {
            if (j == skip_col)
                continue;
            m.data[di][dj] = data[i][j];
            dj++;
        }
        di++;
    }
    return m;
}

// extract a single row as a 1 x cols matrix
Matrix Matrix::get_row(int i) const {
    assert(i >= 0 && i < rows);
    Matrix m(1, cols);
    for (int j = 0; j < cols; j++)
        m.data[0][j] = data[i][j];
    return m;
}

// extract a single column as a rows x 1 matrix
Matrix Matrix::get_col(int j) const {
    assert(j >= 0 && j < cols);
    Matrix m(rows, 1);
    for (int i = 0; i < rows; i++)
        m.data[i][0] = data[i][j];
    return m;
}

// ---- element-wise function application ----

// return a copy with f applied to every entry
Matrix Matrix::apply(const std::function<double(double)> &f) const {
    Matrix m(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m.data[i][j] = f(data[i][j]);
    return m;
}

// apply f to every entry in place
Matrix &Matrix::apply_inplace(const std::function<double(double)> &f) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i][j] = f(data[i][j]);
    return *this;
}

// ---- reductions ----

// sum of all entries
double Matrix::sum() const {
    double s = 0.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            s += data[i][j];
    return s;
}

// sum of the main diagonal, square matrices only
double Matrix::trace() const {
    assert(is_square());
    double s = 0.0;
    for (int i = 0; i < rows; i++)
        s += data[i][i];
    return s;
}

// smallest entry
double Matrix::min() const {
    assert(rows > 0 && cols > 0);
    double v = data[0][0];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            ckmin(v, data[i][j]);
    return v;
}

// largest entry
double Matrix::max() const {
    assert(rows > 0 && cols > 0);
    double v = data[0][0];
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            ckmax(v, data[i][j]);
    return v;
}

// ---- linear algebra ----

// determinant via gaussian elimination with partial pivoting
double Matrix::determinant() const {
    assert(is_square());
    int n = rows;
    vec2<double> a = data; // working copy we are free to mutate
    double det = 1.0;
    for (int col = 0; col < n; col++) {
        // find the row with the largest pivot in this column
        int pivot = col;
        for (int i = col + 1; i < n; i++)
            if (std::fabs(a[i][col]) > std::fabs(a[pivot][col]))
                pivot = i;
        if (a[pivot][col] == 0.0)
            return 0.0; // singular, determinant is zero
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            det = -det; // row swap flips the sign
        }
        det *= a[col][col];
        // eliminate entries below the pivot
        for (int i = col + 1; i < n; i++) {
            double factor = a[i][col] / a[col][col];
            for (int j = col; j < n; j++)
                a[i][j] -= factor * a[col][j];
        }
    }
    return det;
}

// inverse via gauss-jordan elimination with partial pivoting
Matrix Matrix::inverse() const {
    assert(is_square());
    int n = rows;
    vec2<double> a = data;             // working copy of this matrix
    Matrix inv = Matrix::identity(n);  // becomes the inverse
    for (int col = 0; col < n; col++) {
        // pick the largest pivot for numerical stability
        int pivot = col;
        for (int i = col + 1; i < n; i++)
            if (std::fabs(a[i][col]) > std::fabs(a[pivot][col]))
                pivot = i;
        // a zero pivot after partial pivoting means the column is all zero
        if (a[pivot][col] == 0.0)
            throw std::runtime_error("matrix is singular, no inverse");
        if (pivot != col) {
            std::swap(a[pivot], a[col]);
            std::swap(inv.data[pivot], inv.data[col]);
        }
        // scale the pivot row so the pivot becomes 1
        double p = a[col][col];
        for (int j = 0; j < n; j++) {
            a[col][j] /= p;
            inv.data[col][j] /= p;
        }
        // eliminate this column from every other row
        for (int i = 0; i < n; i++) {
            if (i == col)
                continue;
            double factor = a[i][col];
            for (int j = 0; j < n; j++) {
                a[i][j] -= factor * a[col][j];
                inv.data[i][j] -= factor * inv.data[col][j];
            }
        }
    }
    return inv;
}

// raise a square matrix to a non-negative integer power via fast exponentiation
Matrix Matrix::pow(int e) const {
    assert(is_square() && e >= 0);
    Matrix result = Matrix::identity(rows);
    Matrix base = *this;
    while (e > 0) {
        if (e & 1)
            result = result * base;
        base = base * base;
        e >>= 1;
    }
    return result;
}

// ---- output ----

// print rows on separate lines with aligned columns
std::ostream &operator<<(std::ostream &os, const Matrix &m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            if (j)
                os << ' ';
            os << std::setw(10) << m.data[i][j];
        }
        os << '\n';
    }
    return os;
}
