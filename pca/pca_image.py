import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class ProcessImage:
    """
    Apply the function specified processing to an image
    """

    def resize_image(self, image, size: tuple, interpol):
        resized_image = cv2.resize(image, size, interpolation=interpol)
        return resized_image

    def sobel_filter(self, image):
        print("yes")


class ApplyPCA:
    """
    Image compression using PCA.
    """
    def display_channel(self, blue, green, red):
        fig = plt.figure(figsize=(16, 9))
        fig.add_subplot(131)
        plt.title("Blue channel")
        plt.imshow(blue)

        fig.add_subplot(132)
        plt.title("Green channel")
        plt.imshow(green)

        fig.add_subplot(133)
        plt.title("Red channel")
        plt.imshow(red)
        plt.show()

    def variance_plot(self, pca_blue, pca_green, pca_red):
        fig = plt.figure(figsize=(10, 15))
        fig.add_subplot(131)
        plt.bar(list(range(1, 257)), pca_blue.explained_variance_ratio_)
        plt.xlabel("Eigen Value")
        plt.ylabel("Variance")
        plt.title("Blue channel")

        fig.add_subplot(132)
        plt.bar(list(range(1, 257)), pca_green.explained_variance_ratio_)
        plt.xlabel("Eigen Value")
        plt.ylabel("Variance")
        plt.title("Green channel")

        fig.add_subplot(133)
        plt.bar(list(range(1, 257)), pca_red.explained_variance_ratio_)
        plt.xlabel("Eigen Value")
        plt.ylabel("Variance")
        plt.title("Red channel")

        plt.show()

    def display_original_pca(self, original_image, pca_image):
        # plot original image
        fig = plt.figure(figsize=(10, 8))
        fig.add_subplot(121)
        plt.title("Original Image")
        plt.imshow(original_image)

        # plot pca image
        fig.add_subplot(122)
        plt.title("PCA Image")
        plt.imshow(pca_image)

        plt.show()

    def compress_image_pca(self, image, components):
        # split to each channel
        blue, green, red = cv2.split(image)
        # display each channel
        # self.display_channel(blue=blue, green=green, red=red)
        # normalize between 0 and 1
        image = np.float64(image)
        cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
        normalized_blue, normalized_green, normalized_red = cv2.split(image)

        # transforming blue channel
        pca_b = PCA(n_components=components)
        pca_b.fit(normalized_blue)
        pca_norm_blue = pca_b.transform(normalized_blue)
        # transforming green channel
        pca_g = PCA(n_components=components)
        pca_g.fit(normalized_blue)
        pca_norm_green = pca_g.transform(normalized_green)
        # transforming red channel
        pca_r = PCA(n_components=components)
        pca_r.fit(normalized_blue)
        pca_norm_red = pca_r.transform(normalized_red)

        # plotting the variance
        # self.variance_plot(pca_b, pca_g, pca_r)

        # transforming back to original image space
        transformed_blue = pca_b.inverse_transform(pca_norm_blue)
        print(transformed_blue.shape)
        transformed_green = pca_b.inverse_transform(pca_norm_green)
        transformed_red = pca_b.inverse_transform(pca_norm_red)

        # merge channels to form 3 channel image
        pca_image = cv2.merge((transformed_blue, transformed_green, transformed_red))

        # display original and pca images
        self.display_original_pca(original_image=image, pca_image=pca_image)

        # convert image to 0 to 255 scale
        pca_image = cv2.convertScaleAbs(pca_image * 255)

        return pca_image


def main():
    image_path = "image_to_process.jpg"
    img = cv2.imread(image_path)
    resized_img = ProcessImage().resize_image(img, size=(1024, 1024), interpol=cv2.INTER_CUBIC)
    pca_image = ApplyPCA().compress_image_pca(resized_img, 128)
    cv2.imwrite("pca_image.png", pca_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
