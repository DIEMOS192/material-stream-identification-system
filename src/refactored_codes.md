
refacored augmentation code

# def fixed_augment_cycle(image):
#     geoAug = GeometricAugmentor()
#     photoAug = PhotometricAugmentor()
#     image = geoAug.rotate(image)
#     if random.random() < 0.5:
#         image = geoAug.flip(image)
#     if random.random() < 0.3:
#         image = photoAug.adjust_brightness_contrast(image)
#     image = photoAug.Gauss_blur(image)
#     return image

# def adjust_brightness_contrast(self, image):
    
#     brightness = random.uniform(-30, 30)
#     contrast = random.uniform(0.9, 1.1)
#     img = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
#     return img

# def main_v2():
#     root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
#     raw_data_folder = os.path.join(root, 'data', 'raw')
#     aug_data_folder = os.path.join(root, 'data', 'augmented')
#     for category in os.listdir(raw_data_folder):
#         full_size = calculate_training_data_size(os.path.join(raw_data_folder, category))
#         max_size = get_max_data_folder()
#         aug_size = int((max_size - full_size)) + 20
#         if aug_size <= 0:
#             aug_size = 20
#         category_path = os.path.join(raw_data_folder, category)
#         save_category_path = os.path.join(aug_data_folder, category)
#         if not os.path.exists(save_category_path):
#             os.makedirs(save_category_path)
#         imgs = os.listdir(category_path)
#         count = 1
#         attempts = 0
#         while count <= aug_size:
#             attempts += 1
#             if attempts > full_size * 2:
#                 break

#             img = random.choice(imgs)
#             img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
#             image = cv2.imread(os.path.join(category_path, img))
#             if image is None:
#                 continue
#             for i in range(2):
#                 augmented_image = fixed_augment_cycle(image.copy())
#                 if augmented_image is None:
#                     continue
#                 img_name = 'augmented_' + category + '_' + str(count) + '.jpg'
#                 cv2.imwrite(os.path.join(save_category_path, img_name), augmented_image)
#                 count += 1





