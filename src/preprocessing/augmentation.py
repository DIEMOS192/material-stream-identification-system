class GeometricAugmentor:

    def rotate(self):#small degree (±10-25°)
        pass
    def flip(self):
        pass
    def scale(self):#0.9-1.1x
        pass
    def random_crop(self):
        pass
    def random_resize(self):
        pass

class PhotometricAugmentor:

    def adjust_brightness(self):
        pass
    def adjust_contrast(self):
        pass
    def slight_color_jitter(self):
        pass
    def adjust_hue(self):
        pass
    def add_noise(self):#Gaussian, (low variance)
        pass


def choose_random_augmentation(image):
    pass