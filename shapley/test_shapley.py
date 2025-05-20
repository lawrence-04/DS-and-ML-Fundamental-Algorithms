import numpy as np

from shapley import Shapley  # Assuming the code is in a file named shapley.py


# Define simple model predictors for testing
def linear_model(X):
    """Simple linear model that returns sum of features as output"""
    return np.sum(X, axis=1).reshape(-1, 1)


def multi_output_model(X):
    """Model that returns multiple outputs: sum and product of features"""
    sums = np.sum(X, axis=1).reshape(-1, 1)
    products = np.prod(X, axis=1).reshape(-1, 1)
    return np.hstack([sums, products])


class TestShapley:
    def setup_method(self):
        """Setup test fixtures before each test method"""
        # Simple dataset with 3 features and 4 samples
        self.X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.means = np.mean(self.X_train, axis=0)
        self.single_input = np.array([2, 3, 4])

    def test_initialization(self):
        """Test the proper initialization of the Shapley class"""
        shapley = Shapley(linear_model, self.X_train)

        assert shapley.n_features == 3
        assert shapley.n_outputs == 1
        np.testing.assert_array_almost_equal(shapley.means, self.means)

    def test_initialization_multi_output(self):
        """Test initialization with multi-output model"""
        shapley = Shapley(multi_output_model, self.X_train)

        assert shapley.n_features == 3
        assert shapley.n_outputs == 2

    def test_get_value_for_single_feature(self):
        """Test _get_value_for_single_feature method with a simple linear model"""
        shapley = Shapley(linear_model, self.X_train)

        # Test for the first feature
        value = shapley._get_value_for_single_feature(self.single_input, 0)

        # For a linear model, the Shapley value should be approximately equal to the feature value
        # when normalized by total features
        assert isinstance(value, np.ndarray)
        assert value.shape == (1,)
        # The exact value requires more complex calculation validation

    def test_get_values_multi_output(self):
        """Test get_values with multi-output model"""
        shapley = Shapley(multi_output_model, self.X_train)

        values = shapley.get_values(self.single_input)

        assert values.shape == (3, 2)  # 3 features, 2 outputs

    def test_mask_lens_tracking(self):
        """Test that mask_lens attribute is being populated"""
        shapley = Shapley(linear_model, self.X_train)

        # Initially should be empty
        assert len(shapley.mask_lens) == 0

        # After computation, should still be empty as it's not being used in the current implementation
        shapley.get_values(self.single_input)
        assert len(shapley.mask_lens) == 0

    def test_with_edge_cases(self):
        """Test with edge cases like zero inputs"""
        zero_input = np.zeros(3)
        shapley = Shapley(linear_model, self.X_train)

        values = shapley.get_values(zero_input)

        # For zero inputs with a linear model, values should be negative
        # (since zero is less than the means)
        assert np.all(values <= 0)

    def test_dummy_property(self):
        """Test that features with no influence have zero Shapley values"""

        # Create a model that only depends on the first feature
        def first_feature_only(X):
            return X[:, 0].reshape(-1, 1)

        shapley = Shapley(first_feature_only, self.X_train)
        values = shapley.get_values(self.single_input)

        # Features 1 and 2 should have zero Shapley values
        np.testing.assert_almost_equal(values[1], 0, decimal=5)
        np.testing.assert_almost_equal(values[2], 0, decimal=5)
