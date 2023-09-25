# 导入包
import os
import matplotlib.pyplot as plt
import matplotlib.image as mping
# from keras.src.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras import Sequential
# from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# 定义文件夹路径
train_dir = '../Data/train'
test_dir = '../Data/test'


# 查看所有图片数量，使用递归遍历文件夹结构，查看每个子文件夹中图片数量
# 同理可得每个子文件夹各自的数量
def count_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension.lower() in image_extensions:
                image_count += 1

    return image_count


# 指定大文件夹路径
main_folder_path = '../Data/train'
# 调用函数获取图片数量
total_image_count = count_images_in_folder(main_folder_path)

print(f'总共有{total_image_count}张图片')


# 简单显示雏菊(daisy)其中一张图片
def display_image(_image_path):
    img = mping.imread(_image_path)
    img_plot = plt.imshow(img)
    plt.show()


# 指定文件夹路径和图片名称
daisy_path = '../Data/train/雏菊'
image_name = 'image_0806.jpg'

# 构建图片路径
image_path = os.path.join(daisy_path, image_name)

# 调用函数显示图片
display_image(image_path)

# 原始像素值是0-255，为了使模型训练更稳定以及更容易收敛，我们需要标准化数据集，一般来说就是把像素值缩放到0-1
# 在训练集中我们定义了旋转，平移，缩放等功能，这样可以让我们的模型学习旋转、跳跃、闭着眼的花花图片，增加样本使用效率和模型准确度。
# 对数据进行增强
train_datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=20,  # 随机错切变换
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest',  # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale=1 / 255,  # 数据归一化
)

# 使用flow_from_directory从目录中读取图片，生成训练集和测试机的迭代器，用于把图片按一定批次大小传入模型训练
# 设置训练集迭代器
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 训练集存放路径
    target_size=(64, 64),  # 训练集图片尺寸
    batch_size=16,  # 训练集批次
    class_mode='categorical'
)

# 设置测试集迭代器
test_generator = test_datagen.flow_from_directory(
    test_dir,  # 测试集存放路径
    target_size=(64, 64),  # 测试集图片尺寸
    batch_size=16,  # 测试集批次
    class_mode='categorical'
)

# 搭建神经网络
model = Sequential()  # 创建一个神经网络对象

# 添加一个卷积层，传入固定宽高三通道的图片，以32种不同的卷积核构建32张特征图
# 卷积核大小为3*3，构建特征图比例和原图相同，激活函数为relu函数
# model.add()是Keras中用于向模型中添加层的函数
model.add(Conv2D(input_shape=(64, 64, 3), filters=32, kernel_size=3, padding='same', activation='relu'))
# 再次构建一个卷积层
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

# 构建一个池化层，提取特征，池化层的池化窗口为2*2，步长为2
model.add(MaxPool2D(pool_size=2, strides=2))

# 继续构造卷积层和池化层，区别是卷积核数量为64
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
# 继续构造卷积层和池化层，区别是卷积核数量为64
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten())  # 数据扁平化
model.add(Dense(units=128, activation='relu'))  # 构建一个具有128个神经元的全连接层
model.add(Dense(units=64, activation='relu'))  # 构建一个具有64个神经元的全连接层
DROPOUT_RATE = 0.5
model.add(Dropout(DROPOUT_RATE))  # 加入dropout，防止过拟合
# CLASS = 17
model.add(Dense(units=len(train_generator.class_indices), activation='softmax'))  # 输出层，一共17个神经元，对应17个分类

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# model.fit(train_generator, epochs=100, validation_data=test_generator)

# 保存模型
# model.save('flower_classifier.h5')

# 加载模型
loaded_model = load_model('flower_classifier.h5')


# 预测花朵类别的函数
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255  # 归一化，模型期望值是[0，1]之间的浮点数

    # 使用模型进行预测
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    print(predictions)

    # 获取标签
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    return labels[predicted_class]


# 测试模型
test_image_path = "../Data/train/雏菊/image_0822.jpg"
predict_class = predict_flower(test_image_path)
print(f'The image is predicted as : {predict_class}')
