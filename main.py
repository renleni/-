import numpy as np
from PIL import Image
from scipy.stats import kurtosis, boxcox
from scipy.special import inv_boxcox

def generate_image_with_kurtosis(target_kurtosis, image_size=(500, 500), lambda_start=0.0, lambda_step=0.01):
    """
    生成一个偏度为0，峰度为目标值的灰度图像
    
    :param target_kurtosis: 目标峰度值
    :param image_size: 图像尺寸，默认为500x500
    :param lambda_start: Box-Cox变换的初始lambda值
    :param lambda_step: lambda值的步长
    :return: None，保存名为'custom_kurtosis_image.png'的图像
    """
    # 生成符合要求的随机数据
    data = np.random.normal(loc=0, scale=1, size=image_size)
    
    # Box-Cox变换需要数据为正数，因此将数据平移到正区域
    data_min = data.min()
    data_shifted = data - data_min + 1e-6
    
    # 通过调整lambda值来找到接近目标峰度的变换
    lambda_value = lambda_start
    current_kurtosis = kurtosis(boxcox(data_shifted, lmbda=lambda_value).flatten())
    while abs(current_kurtosis - target_kurtosis) > 0.01:
        lambda_value += lambda_step
        transformed_data = boxcox(data_shifted, lmbda=lambda_value)
        current_kurtosis = kurtosis(transformed_data.flatten())
    
    # 将变换后的数据缩放到0-65535的范围，以适应16位灰度图像
    transformed_data = transformed_data - transformed_data.min()
    transformed_data = transformed_data / transformed_data.max()
    transformed_data = np.round(transformed_data * 44).astype(np.uint8)
    
    # 计算图像的峰度
    transformed_kurtosis = kurtosis(transformed_data.flatten())
    print('生成的图像的峰度为：', transformed_kurtosis)
    
    # 保存图像
    img = Image.fromarray(transformed_data)
    img_name = 'custom_kurtosis_image_{}.png'.format(str(target_kurtosis).replace('.', '_'))
    img.save(img_name)

# 生成峰度为2的图像
generate_image_with_kurtosis(target_kurtosis=0.8)
