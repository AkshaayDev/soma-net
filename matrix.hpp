#ifndef SOMA_MATRIX_HPP
#define SOMA_MATRIX_HPP

#include <vector>
#include <iostream>
#include <functional>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace Soma {

// Minimal matrix class
class Matrix {
public:
	std::vector<std::vector<double>> data;

	// Constructors
	Matrix() {}
	Matrix(std::vector<std::vector<double>> data) : data(data) {}
	Matrix(int rows, int cols) {
		this->resize(rows, cols);
	}

	// Getters
	int rows() const { return data.size(); }
	int cols() const { return data.empty() ? 0 : data[0].size(); }

	// Helpers
	// Print the matrix
	static void print(Matrix& m) {
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.cols(); j++) {
				std::cout << m[i][j] << " ";
			}
			std::cout << "\n";
		}
	}
	// Check whether two matrices are of the same size
	inline static bool sameSize(Matrix a, Matrix b) {
		return (a.rows() == b.rows()) && (a.cols() == b.cols());
	}
	// Return a column matrix (nx1) given a flattened vector
	static Matrix fromVector(std::vector<double> vec) {
		Matrix res(vec.size(), 1);
		for (int i = 0; i < vec.size(); i++) {
			res[i][0] = vec[i];
		}
		return res;
	}
	// Return a scalar matrix (1x1) given a single scalar
	static Matrix fromScalar(double scalar) {
		Matrix res;
		res.data = {{scalar}};
		return res;
	}
	// Resize the number of rows and columns
	void resize(int rows, int cols) {
		data.resize(rows);
		for (int i = 0; i < rows; i++) {
			data[i].resize(cols);
		}
	}
	// Apply a function to each element of the matrix with its value, row and column
	void forEach(const std::function<void(double*, int, int)>& func) {
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				func(&data[i][j], i, j);
			}
		}
	}
	// Fill the matrix with a given value
	void fill(double value) {
		forEach([value](double *val, int, int) {
			*val = value;
		});
	}
	// Check whether the matrix has a nan
	inline bool hasNan() {
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				if (std::isnan(data[i][j])) return true;
			}
		}
		return false;
	}

	// Operations
	// Scalar Addition (Matrix + Scalar)
	Matrix operator+(double scalar) const {
		Matrix res = *this;
		res.forEach([scalar](double *val, int, int) {
			*val += scalar;
		});
		return res;
	}
	// Scalar Addition (Scalar + Matrix)
	friend Matrix operator+(double scalar, Matrix mat) {
		return mat + scalar;
	}
	// Element-wise Addition (Matrix + Matrix)
	Matrix operator+(const Matrix& other) const {
		if (!Matrix::sameSize(*this, other)) throw std::runtime_error("Matrix addition dimension mismatch: " +
			std::to_string(rows()) + "x" + std::to_string(cols()) + " + " +
			std::to_string(other.rows()) + "x" + std::to_string(other.cols())
		);
		Matrix res = *this;
		res.forEach([&other](double *val, int i, int j) {
			*val += other[i][j];
		});
		return res;
	}
	// Unary Negation (-Matrix)
	Matrix operator-() const {
		return *this * -1;
	}
	// Scalar Subtraction (Matrix - Scalar)
	Matrix operator-(double scalar) const {
		Matrix res = *this;
		res.forEach([scalar](double *val, int, int) {
			*val -= scalar;
		});
		return res;
	}
	// Scalar Subtraction (Scalar - Matrix)
	friend Matrix operator-(double scalar, Matrix mat) {
		return scalar + -mat;
	}
	// Element-wise Subtraction (Matrix - Matrix)
	Matrix operator-(const Matrix& other) const {
		if (!Matrix::sameSize(*this, other)) throw std::runtime_error("Matrix subtraction dimension mismatch: " +
			std::to_string(rows()) + "x" + std::to_string(cols()) + " - " +
			std::to_string(other.rows()) + "x" + std::to_string(other.cols())
		);
		Matrix res = *this;
		res.forEach([&other](double *val, int i, int j) {
			*val -= other[i][j];
		});
		return res;
	}
	// Scalar Multiplication (Matrix * Scalar)
	Matrix operator*(double scalar) const {
		Matrix res = *this;
		res.forEach([scalar](double *val, int, int) {
			*val *= scalar;
		});
		return res;
	}
	// Scalar Multiplication (Scalar * Matrix) 
	friend Matrix operator*(double scalar, Matrix mat) {
		mat.forEach([scalar](double *val, int, int) {
			*val *= scalar;
		});
		return mat;
	}
	// Element-wise Multiplication (Matrix * Matrix)
	Matrix operator*(const Matrix& other) const {
		if (!Matrix::sameSize(*this, other)) throw std::runtime_error("Matrix multiplication dimension mismatch: " +
			std::to_string(rows()) + "x" + std::to_string(cols()) + " * " +
			std::to_string(other.rows()) + "x" + std::to_string(other.cols())
		);
		Matrix res = *this;
		res.forEach([&other](double *val, int i, int j) {
			*val *= other[i][j];
		});
		return res;
	}
	// Scalar Division (Matrix / Scalar) 
	Matrix operator/(double scalar) const {
		if (scalar == 0) throw std::runtime_error("Cannot divide matrix by 0");
		Matrix res = *this;
		res.forEach([scalar](double *val, int, int) {
			*val /= scalar;
		});
		return res;
	}
	// Scalar Division (Scalar / Matrix)
	friend Matrix operator/(double scalar, Matrix mat) {
		mat.forEach([scalar](double *val, int, int) {
			if (*val == 0) throw std::runtime_error("Cannot divide scalar by 0 element");
			*val = scalar / *val;
		});
		return mat;
	}
	// Element-wise Division (Matrix / Matrix)
	Matrix operator/(const Matrix& other) const {
		if (!Matrix::sameSize(*this, other)) throw std::runtime_error("Matrix division dimension mismatch: " +
			std::to_string(rows()) + "x" + std::to_string(cols()) + " / " +
			std::to_string(other.rows()) + "x" + std::to_string(other.cols())
		);
		Matrix res = *this;
		res.forEach([&other](double *val, int i, int j) {
			if (other[i][j] == 0) throw std::runtime_error("Cannot element-wise divide by 0");
			*val /= other[i][j];
		});
		return res;
	}
	// Scalar Exponent (Matrix ^ Scalar)
	Matrix operator^(double scalar) const {
		Matrix res = *this;
		res.forEach([scalar](double *val, int, int) {
			*val = std::pow(*val, scalar);
		});
		return res;
	}
	// Scalar Exponent (Scalar ^ Matrix)
	friend Matrix operator^(double scalar, Matrix mat) {
		mat.forEach([scalar](double *val, int, int) {
			*val = std::pow(scalar, *val);
		});
		return mat;
	}
	// Access data directly (modifiable)
	std::vector<double>& operator[](int row) {
		return data[row];
	}
	// Access data directly (read-only)
	const std::vector<double>& operator[](int row) const {
		return data[row];
	}
	// Dot product of two matrices
	static Matrix dot(const Matrix& a, const Matrix& b) {
		if (a.cols() != b.rows()) {
			throw std::runtime_error("Matrix dot product dimension mismatch: " +
				std::to_string(a.rows()) + "x" + std::to_string(a.cols()) + " . " +
				std::to_string(b.rows()) + "x" + std::to_string(b.cols())
			);
		}
		Matrix result(a.rows(), b.cols());
		result.fill(0);
		for (int i = 0; i < a.rows(); i++) {
			for (int j = 0; j < b.cols(); j++) {
				for (int k = 0; k < b.rows(); k++) {
					result[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		return result;
	}
	// Transpose the matrix (Switch rows and columns)
	Matrix transpose() const {
		Matrix res(cols(), rows());
		for (int i = 0; i < rows(); i++) {
			for (int j = 0; j < cols(); j++) {
				res[j][i] = data[i][j];
			}
		}
		return res;
	}
	// Get the maximum value from the matrix
	double max() {
		double max = data[0][0];
		forEach([&max](double *val, int, int) {
			max = std::max(*val, max);
		});
		return max;
	}
	// Get the sum of all elements in the matrix
	double sum() {
		double sum = 0;
		forEach([&sum](double *val, int, int) {
			sum += *val;
		});
		return sum;
	}
};

}

#endif
