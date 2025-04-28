# SCCD-IS
from PIL import Image
import numpy as np
import os

# Gamma correction function (gamma value can be adjusted based on actual measurements)
def gamma_correction(R, G, B):
    LR = R ** 2.1
    LG = G ** 2.1
    LB = B ** 2.1
    return LR, LG, LB

# Inverse gamma correction function
def inverse_gamma_correction(LR, LG, LB):
    R = LR ** (1 / 2.1)
    G = LG ** (1 / 2.1)
    B = LB ** (1 / 2.1)
    return np.clip(R, 0, 255), np.clip(G, 0, 255), np.clip(B, 0, 255)

# Calculate average brightness of an image
def calculate_brightness(image):
    data = np.array(image)
    brightness = np.apply_along_axis(lambda x: gamma_correction(*x), 2, data)  # Apply gamma correction to each pixel
    return np.mean(brightness, axis=(0, 1))  # Calculate average brightness per channel

# New brightness adjustment function
def adjust_brightness(channel_data, original_channel_data, single_image_brightness, overall_brightness):
    adjusted_data = np.copy(channel_data).astype(np.float64)
    print(f"Adjusting brightness for channel where single_image_brightness: {single_image_brightness} and overall_brightness: {overall_brightness}")

    pixel_count = adjusted_data.size
    brightness_diff = overall_brightness - single_image_brightness
    total_adjustment = brightness_diff * pixel_count

    if total_adjustment > 0:
        # When increasing brightness, exclude pixels with value 255
        mask = original_channel_data < 225  # Create mask to exclude 255-value pixels
        valid_pixels = mask.sum()  # Count pixels to adjust
        if valid_pixels == 0:
            return np.clip(channel_data, 0, 255)  # Return original data if no pixels need adjustment

        # Calculate adjustment distribution
        adjustment_ratio = (original_channel_data / original_channel_data.max()) ** 0.5  # Use squared original RGB values for adjustment ratio
        adjustment_ratio[~mask] = 0  # Set adjustment ratio to 0 for excluded pixels
        increase_factor = total_adjustment / np.sum(adjustment_ratio)  # Calculate adjustment factor
        adjusted_data += increase_factor * adjustment_ratio  # Apply adjustment

    elif total_adjustment < 0:
        # When decreasing brightness, exclude pixels with value 0-30
        mask = original_channel_data > 0  # Create mask to exclude 0-30 value pixels
        valid_pixels = mask.sum()  # Count pixels to adjust
        if valid_pixels == 0:
            return np.clip(channel_data, 0, 255)  # Return original data if no pixels need adjustment

        # Calculate adjustment distribution
        adjustment_ratio = (original_channel_data / original_channel_data.max()) ** 2.1  # Use original RGB values for adjustment ratio
        adjustment_ratio[~mask] = 0  # Set adjustment ratio to 0 for excluded pixels
        decrease_factor = (-total_adjustment) / np.sum(adjustment_ratio)  # Calculate adjustment factor
        adjusted_data -= decrease_factor * adjustment_ratio  # Apply adjustment

    print(f"Adjusted channel data range: {adjusted_data.min()} to {adjusted_data.max()}")
    return adjusted_data

# Batch process images
def process_images(image_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate per-image brightness and overall average brightness
    images_brightness = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            data = np.array(image)

            # Add 1 to pixels with RGB value 0
            data[data == 0] += 1  # Use boolean indexing to increment only 0-value pixels

            # Calculate brightness using modified data
            brightness = calculate_brightness(Image.fromarray(data))
            images_brightness.append(brightness)

    overall_brightness = np.mean(np.array(images_brightness), axis=0)

    # Adjust brightness for each image and save results
    for filename, image_brightness in zip(os.listdir(image_folder), images_brightness):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert("RGB")
            data = np.array(image)

            # Add 1 to pixels with RGB value 0
            data[data == 0] += 1  # Use boolean indexing to increment only 0-value pixels

            gamma_corrected_data = np.apply_along_axis(lambda x: gamma_correction(*x), 2, data)  # Apply gamma correction per pixel

            # Adjust brightness for each channel
            adjusted_channels = []
            for i in range(3):  # Iterate through R, G, B channels
                channel_data = gamma_corrected_data[:, :, i]  # Gamma-corrected channel data
                original_channel_data = data[:, :, i]  # Original RGB channel data
                adjusted_channel = adjust_brightness(channel_data, original_channel_data, image_brightness[i], overall_brightness[i])
                adjusted_channels.append(adjusted_channel)

            # Recombine adjusted channels into RGB image
            adjusted_data = np.stack(adjusted_channels, axis=2)
            adjusted_data = np.apply_along_axis(lambda x: inverse_gamma_correction(*x), 2, adjusted_data)  # Apply inverse gamma correction
            adjusted_data = np.clip(adjusted_data, 0, 255).astype(np.uint8)  # Ensure values are within 0-255

            # Save adjusted image
            adjusted_image = Image.fromarray(adjusted_data)
            output_path = os.path.join(output_folder, filename)
            adjusted_image.save(output_path)

    print("All images have been processed and saved to the output folder.")

# Input and output folder paths
image_folder = r"input path"
output_folder = r"output path"

# Execute batch processing
process_images(image_folder, output_folder)
