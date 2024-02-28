import numpy as np
import cv2
import matplotlib.pyplot as plt

def amef(I_hazy, clip_range):
    I_hazy = I_hazy.astype(np.float32) / 255.0
    height, width, channels = I_hazy.shape
    I = np.zeros((height, width, channels, 6), dtype=np.float32)

    I[:, :, :, 0] = I_hazy
    range_values = np.linspace(1, 6, 6)

    # Gamma 校正
    for i in range(1, 5):
        gamma = range_values[i]
        for c in range(channels):
            I[:, :, c, i] = np.power(I_hazy[:, :, c], gamma)

    # CLAHE
    for c in range(channels):
        clahe = cv2.createCLAHE(clipLimit=clip_range)
        I_hazy_channel = (I_hazy[:, :, c] * 255).astype(np.uint8)
        I[:, :, c, 5] = ((clahe.apply(I_hazy_channel)).astype(np.float32)) / 255.0
    print("I的shape :",I.shape)    

    # for i in range(6):
    #    plt.subplot(1, 6, i+1)
    #    plt.imshow(cv2.cvtColor(I[:,:,:,i], cv2.COLOR_BGR2RGB))
    #    plt.title(f'Gamma {i + 1}')
    # plt.show()

    R = exposure_fusion(I)

    return R

def exposure_fusion(I):
    r, c, _, N = I.shape

    # 將對比度和飽和度結合為權重圖
    W = np.ones((r, c, 3, N), dtype=np.float32) * contrast(I) * saturation(I)
    print("W 原本的shape :",W.shape)

    # for i in range(W.shape[3]):
    #    plt.subplot(1, W.shape[3], i+1)
    #    plt.imshow(W[:,:,0,i], cmap='gray')
    #    plt.title(f'W {i + 1}')
    # plt.show()

    # 加個eps避免除0 & 歸一化 
    W = W + 1e-12
    W = W / np.tile(np.sum(W, axis=3)[:, :, :, np.newaxis], (1, 1, 1, N))   #或是這樣也可以 W = W / (np.sum(W, axis=2, keepdims=True))
    print("W 歸一化後的shape :",W.shape)
    print("W.dtype :",W.dtype)

    # for i in range(W.shape[3]):
    #    plt.subplot(1, W.shape[3], i+1)
    #    plt.imshow(W[:,:,0,i], cmap='gray')
    #    plt.title(f'歸一化後W {i + 1}')
    # plt.show()

    # 創建一個空金字塔
    pyr = gaussian_pyramid(np.zeros((r, c, 3), dtype=np.float32))
    nlev = len(pyr)

    # 多分辨率融合
    for i in range(N):
        # 從每個輸入圖像構建金字塔
        pyrW = gaussian_pyramid(W[:, :, :, i])
        pyrI = laplacian_pyramid(I[:, :, :, i])

        # 融合
        for l in range(nlev):
            pyr[l] = pyr[l] + pyrW[l] * pyrI[l]

    # for i in range(len(pyr)):
    #    plt.subplot(1, len(pyr), i+1)
    #    plt.imshow(cv2.cvtColor((np.clip(pyr[i], 0, 1)*255).astype(np.float32),cv2.COLOR_BGR2RGB))
    # #    cv2.imshow("111211", (np.clip(pyr[i], 0, 1)*255).astype(np.float32))
    # #    cv2.waitKey(0)
    #    plt.title(f'Level {i + 1}')
    # plt.show()

    # 重建
    R = reconstruct_laplacian_pyramid(pyr)
    return R

def contrast(I):
    h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)  # Laplacian filter
    N = I.shape[3]
    C = np.zeros((I.shape[0], I.shape[1], 3, N), dtype=np.float32)

    for i in range(N):
        mono = cv2.cvtColor(I[:, :, :, i], cv2.COLOR_BGR2GRAY)
        mono = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        C[:, :, :, i] = np.abs(cv2.filter2D(mono, -1, h, borderType=cv2.BORDER_REPLICATE))
        print(f"對比度第{i+1}張圖的shape :",C.shape)
    print("contrast C.dtype :",C.dtype)

    # for i in range(C.shape[3]):
    #    plt.subplot(1, C.shape[3], i+1)
    #    plt.imshow(C[:,:,0,i], cmap='gray')
    #    plt.title(f'contrast {i + 1}')
    # plt.show()

    return C

def saturation(I):
    N = I.shape[3]
    C = np.zeros((I.shape[0], I.shape[1], 3, N), dtype=np.float32)

    for i in range(N):
        # 飽和度計算為顏色通道的標準差
        R = I[:, :, 0, i]
        G = I[:, :, 1, i]
        B = I[:, :, 2, i]
        mu = (R + G + B) / 3
        sat = np.sqrt(((R - mu)**2 + (G - mu)**2 + (B - mu)**2) / 3)
        sat = cv2.cvtColor(sat, cv2.COLOR_GRAY2BGR)
        C[:, :, :, i] = sat
        print(f"飽和度第{i+1}張圖的shape :",C.shape)
    print("saturation C.dtype :",C.dtype)

    # for i in range(C.shape[3]):
    #    plt.subplot(1, C.shape[3], i+1)
    #    plt.imshow(C[:,:,0,i], cmap='gray')
    #    plt.title(f'saturation {i + 1}')
    # plt.show()
        
    return C

def pyramid_filter():
    return np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
filter = pyramid_filter()

def downsample(I, filter):
    # 與自定義 filter 進行卷積
    R = cv2.filter2D(I, -1, np.expand_dims(filter,axis=0), borderType=cv2.BORDER_REFLECT)     # 水平
    R = cv2.filter2D(R, -1, np.expand_dims(filter,axis=1), borderType=cv2.BORDER_REFLECT)     # 垂直

    # 下採樣
    R = R[::2, ::2, :]
    return R

def upsample(I, odd, filter):
    # 增加分辨率
    I = np.pad(I, [(1, 1), (1, 1), (0, 0)], mode='edge')  # 用1像素邊界填充圖像
    r, c, k = 2 * I.shape[0], 2 * I.shape[1], I.shape[2]
    R = np.zeros((r, c, k))

    R[0:r:2, 0:c:2, :] = 4 * I                        # R[0:r:2, 0:c:2, :] 和 R[::2, ::2, :] 是一樣意思的
                                                      # 4 * I 可以看作是一種插值方式，將原始像素的值擴展到更大的區域，
                                                      #    以維持亮度或顏色的一致性，這方法可以幫助在上採樣過程中減少亮度或顏色的損失。
    # 插值，與可分離濾波器進行卷積
    R = cv2.filter2D(R, -1, np.expand_dims(filter,axis=0), borderType=cv2.BORDER_REFLECT)   # 就算不指定填充邊界的方式，cv2.filter2D默認使用cv2.BORDER_CONSTANT
    R = cv2.filter2D(R, -1, np.expand_dims(filter,axis=1), borderType=cv2.BORDER_REFLECT)

    # 刪除邊界
    R = R[2:r-2-odd[0], 2:c-2-odd[1], :]
    return R

def gaussian_pyramid(I, nlev=None):
    r = I.shape[0]
    c = I.shape[1]

    if nlev is None:
        # 計算金字塔最高層數
        nlev = int(np.floor(np.log2(min(r, c))))
    print(f"金字塔總共有{nlev}層")

    # 創建包含 nlev 個元素的全None列表，第0層=原始影像
    pyr = [None] * nlev
    pyr[0] = I.copy()

    # 遞迴式的下採樣
    for l in range(1, nlev):
        pyr[l] = downsample(pyr[l-1], filter)

    # for i in range(len(pyr)):
    #    plt.subplot(1, len(pyr), i+1)
    #    plt.imshow(pyr[i][:,:,0], cmap='gray')
    #    plt.title(f'Level {i + 1}')
    # plt.show()

    return pyr

def laplacian_pyramid(I, nlev=None):
    r = I.shape[0]
    c = I.shape[1]

    if nlev is None:
        # 計算金字塔最高層數
        nlev = int(np.floor(np.log2(min(r, c))))
    print(f"金字塔總共有{nlev}層")

    # 遞迴建立金字塔
    pyr = [None] * nlev
    filter = pyramid_filter()
    J = I.copy()

    for l in range(nlev - 1):
        # 應用低通濾波器，然後下採樣
        I = downsample(J, filter)
        odd = (2 * I.shape[0] - J.shape[0], 2 * I.shape[1] - J.shape[1])  # 檢查上採樣版本是否需要奇數
        # 在每一層中，存儲圖像和上採樣低通版本之間的差異
        pyr[l] = J - upsample(I, odd, filter)

        J = I  # 繼續使用低通圖像

    pyr[nlev - 1] = J  # 最粗糙的層包含剩餘的低通圖像

    # for i in range(len(pyr)):
    #    plt.subplot(1, len(pyr), i+1)
    #    plt.imshow(cv2.cvtColor((((np.clip(pyr[i], 0, 1))*255).astype(np.float32)),cv2.COLOR_BGR2RGB))
    # #    plt.imshow(np.clip(cv2.cvtColor(((pyr[i])*255).astype(np.uint8),cv2.COLOR_BGR2RGB), 0, 255))   這是錯的 要先截斷 再*255 再轉uint8 才對
    #    plt.title(f'Level {i + 1}')
    # plt.show()

    return pyr

def reconstruct_laplacian_pyramid(pyr):
    nlev = len(pyr)
    r, c, _ = pyr[0].shape
    print("reconstruct_laplacian_pyramid r,c :",(r,c))
    R = pyr[nlev - 1].copy()
    filter = pyramid_filter()

    for l in range(nlev - 2, -1, -1):
        odd = (2 * R.shape[0] - pyr[l].shape[0], 2 * R.shape[1] - pyr[l].shape[1])
        R = pyr[l] + upsample(R, odd, filter)
    R = R.astype(np.float32)
    print("重建回傳的R.dtype :", R.dtype)
    return R



aa = cv2.imread("dde.png")
result = amef(aa, clip_range = 0.1)

result_255 = np.clip(result, 0, 1)
result_255 = result_255*255
result_255 = result_255.astype(np.uint8)

print("aa.dtype     :", aa.dtype)
print("result.dtype :", result.dtype)

print("aa max :", np.max(aa))
print("result max :", np.max(result_255))


# 顯示原始圖像和處理後的結果
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(aa,cv2.COLOR_RGB2BGR))
plt.title('Original Image')
#plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result_255,cv2.COLOR_RGB2BGR))
plt.title('Result Image')
#plt.axis('off')

plt.show()

#cv2.imwrite("result_255.png", result_255)