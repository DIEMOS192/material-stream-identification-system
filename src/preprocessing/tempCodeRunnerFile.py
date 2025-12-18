class FeatureExtractor:
    
    def flatten_image(self, image):
        image = cv2.resize(image, (64, 64))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.flatten()
    
    def extract_color_histogram(self, image):
        pixel_bins = [14, 12, 12]  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, pixel_bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        features = hog(gray, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), 
                        orientations=9, 
                        block_norm='L2-Hys',
                        visualize=False)
        return features