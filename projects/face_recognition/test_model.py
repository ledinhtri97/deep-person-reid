from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='resnet50_fc512',
    model_path='resutls/resnet50_market1501_triplet1/model/model.pth',
    device='cuda'
)

image_list = [
    'f0003_s0.932_face_3065.jpg',
]

features = extractor(image_list)
print(features.shape) # output (5, 512)