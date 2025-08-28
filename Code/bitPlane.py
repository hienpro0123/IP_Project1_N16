import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10 # cỡ chữ mặc định 10pt
img = cv2.imread(r"img11.tif",0)

lst = []
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        lst.append(np.binary_repr(img[i][j],width=8)) # width số bit
# Đã có 1 danh sách chứa chuỗi các bit
# Để tạo ra các mặt phẳng bit chúng ta lặp qua các phần tử chuỗi các bit
# Rút trích nó là lưu vào từng danh sách (mỗi danh sách là mặt phẳng bit)

#Rút trích thành mặt phẳng bit thứ 8
mp_bit_8 =[]
for i in lst:
    mp_bit_8.append(int(i[0]))

#Rút trích thành mặt phẳng bit thứ 7
mp_bit_7 =[]
for i in lst:
    mp_bit_7.append(int(i[1]))

#Rút trích thành mặt phẳng bit thứ 6
mp_bit_6 =[]
for i in lst:
    mp_bit_6.append(int(i[2]))

#Rút trích thành mặt phẳng bit thứ 5
mp_bit_5 =[]
for i in lst:
    mp_bit_5.append(int(i[3]))

#Rút trích thành mặt phẳng bit thứ 4
mp_bit_4 =[]
for i in lst:
    mp_bit_4.append(int(i[4]))

#Rút trích thành mặt phẳng bit thứ 3
mp_bit_3 =[]
for i in lst:
    mp_bit_3.append(int(i[5]))

#Rút trích thành mặt phẳng bit thứ 2
mp_bit_2 =[]
for i in lst:
    mp_bit_2.append(int(i[6]))

#Rút trích thành mặt phẳng bit thứ 1
mp_bit_1 =[]
for i in lst:
    mp_bit_1.append(int(i[7]))

# Tái tạo ảnh
# Nhân từng phần tử với 2^(n-1) để có ảnh tái tạo tương ứng với mỗi mặt phẳng bit
# n là bit thứ n
image_bit_8 = (np.array(mp_bit_8,dtype='uint8')*128).reshape(img.shape[0],img.shape[1])
image_bit_7 = (np.array(mp_bit_7,dtype='uint8')*64).reshape(img.shape[0],img.shape[1])
image_bit_6 = (np.array(mp_bit_6,dtype='uint8')*32).reshape(img.shape[0],img.shape[1])
image_bit_5 = (np.array(mp_bit_5,dtype='uint8')*16).reshape(img.shape[0],img.shape[1])
image_bit_4 = (np.array(mp_bit_4,dtype='uint8')*8).reshape(img.shape[0],img.shape[1])
image_bit_3 = (np.array(mp_bit_3,dtype='uint8')*4).reshape(img.shape[0],img.shape[1])
image_bit_2 = (np.array(mp_bit_2,dtype='uint8')*2).reshape(img.shape[0],img.shape[1])
image_bit_1 = (np.array(mp_bit_1,dtype='uint8')*1).reshape(img.shape[0],img.shape[1])

#Tạo vùng vẽ tỷ lệ 16:9
fig = plt.figure(figsize=(16, 9))
#Tạo 9 vùng vẽ con, phân bố 3 hàng 3 cột
(ax1, ax2, ax3), (ax4,ax5, ax6),(ax7, ax8, ax9) = fig.subplots(3, 3)

#Vẽ ảnh gốc
ax1.imshow(img, cmap='gray')
ax1.set_title("ảnh gốc")

#Vẽ ảnh bit thứ 8
ax2.imshow(image_bit_8, cmap='gray')
ax2.set_title("Mặt phẳng bit 8")

#Vẽ ảnh bit thứ 7
ax3.imshow(image_bit_7, cmap='gray')
ax3.set_title("Mặt phẳng bit 7")

#Vẽ ảnh bit thứ 6
ax4.imshow(image_bit_6, cmap='gray')
ax4.set_title("Mặt phẳng bit 6")

#Vẽ ảnh bit thứ 5
ax5.imshow(image_bit_5, cmap='gray')
ax5.set_title("Mặt phẳng bit 5")

#Vẽ ảnh bit thứ 4
ax6.imshow(image_bit_4, cmap='gray')
ax6.set_title("Mặt phẳng bit 4")

#Vẽ ảnh bit thứ 3
ax7.imshow(image_bit_3, cmap='gray')
ax7.set_title("Mặt phẳng bit 3")

#Vẽ ảnh bit thứ 2
ax8.imshow(image_bit_2, cmap='gray')
ax8.set_title("Mặt phẳng bit 2")

#Vẽ ảnh bit thứ 1
ax9.imshow(image_bit_1, cmap='gray')
ax9.set_title("Mặt phẳng bit 1")
plt.show() # Hiển thị vùng vẽ ảnh

# Tái tạo ảnh từ việc kết hợp nhiều mặt phẳng bit
#Tạo vùng vẽ tỷ lệ 16:9
fig1 = plt.figure(figsize=(16, 9))
#Tạo 4 vùng vẽ con, phân bố 2 hàng 2 cột
(bx1, bx2), (bx3, bx4) = fig1.subplots(2, 2)
#Vẽ ảnh gốc
bx1.imshow(img, cmap='gray')
bx1.set_title("ảnh gốc")

#Vẽ ảnh bit thứ 8 và bít thứ 7
bx2.imshow(image_bit_8 + image_bit_7, cmap='gray')
bx2.set_title("Mặt phẳng bit 8 và bit 7")

#Vẽ ảnh bit thứ 8, bít thứ 7 và bít thứ 6
bx3.imshow(image_bit_8 + image_bit_7+ image_bit_6, cmap='gray')
bx3.set_title("Mặt phẳng bit 8, bit 7 và bit 6")

#Vẽ ảnh bit thứ 7, bít thứ 6 và bít thứ 5
bx4.imshow(image_bit_7 + image_bit_6+ image_bit_5, cmap='gray')
bx4.set_title("Mặt phẳng bit 7, bit 6 và bit 5")

plt.show() # Hiển thị vùng vẽ ảnh
