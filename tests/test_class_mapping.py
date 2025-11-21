# """
# Test cases for class mapping correctness
# """
# import pytest
# import tensorflow as tf
# from ChestCancerClassifier.pipeline.prediction import PredictionPipeline


# class TestClassMapping:
#     """Test cases to verify class mapping is correct"""

#     def test_class_mapping_structure(self):
#         """Test that class mapping has correct structure"""
#         pipeline = PredictionPipeline("test.jpg")
#         mapping = pipeline.class_mapping
        
#         assert isinstance(mapping, dict)
#         assert 0 in mapping
#         assert 1 in mapping
#         assert len(mapping) == 2

#     def test_class_0_is_cancer(self):
#         """Test that class 0 maps to Adenocarcinoma Cancer"""
#         pipeline = PredictionPipeline("test.jpg")
#         assert pipeline.class_mapping[0] == 'Adenocarcinoma Cancer'

#     def test_class_1_is_normal(self):
#         """Test that class 1 maps to NORMAL"""
#         pipeline = PredictionPipeline("test.jpg")
#         assert pipeline.class_mapping[1] == 'NORMAL'

#     def test_class_mapping_matches_training(self):
#         """Test that class mapping matches ImageDataGenerator ordering"""
#         # ImageDataGenerator orders classes alphabetically
#         # 'adenocarcinoma' comes before 'normal'
#         # So: adenocarcinoma = 0, normal = 1
        
#         gen = tf.keras.preprocessing.image.ImageDataGenerator()
#         # This would normally be called with a directory, but we're just
#         # verifying the expected alphabetical ordering
        
#         # Verify our mapping matches expected alphabetical order
#         pipeline = PredictionPipeline("test.jpg")
#         assert pipeline.class_mapping[0] == 'Adenocarcinoma Cancer'
#         assert pipeline.class_mapping[1] == 'NORMAL'

#     @pytest.mark.parametrize("class_idx,expected_label", [
#         (0, 'Adenocarcinoma Cancer'),
#         (1, 'NORMAL')
#     ])
#     def test_class_mapping_values(self, class_idx, expected_label):
#         """Parametrized test for class mapping values"""
#         pipeline = PredictionPipeline("test.jpg")
#         assert pipeline.class_mapping[class_idx] == expected_label

