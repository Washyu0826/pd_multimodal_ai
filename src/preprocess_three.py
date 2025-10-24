"""
帕金森氏症手寫數據預處理工具包
包含最新的去噪、增強、骨架化和數據增強技術
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, filters, exposure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import Tuple, List
import albumentations as A


class PDHandwritingPreprocessor:
    """PD手寫數據預處理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化預處理器
        
        Args:
            target_size: 目標圖像尺寸 (height, width)
        """
        self.target_size = target_size
        
    def load_image(self, image_path: str) -> np.ndarray:
        """載入圖像"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"無法載入圖像: {image_path}")
        return img
    
    def non_local_means_denoise(self, img: np.ndarray, 
                                h: int = 10, 
                                template_window: int = 7,
                                search_window: int = 21) -> np.ndarray:
        """
        非局部均值去噪 (最新技術)
        比高斯濾波更能保留筆劃細節
        
        Args:
            img: 輸入圖像
            h: 濾波強度 (越大去噪越強，但可能過度平滑)
            template_window: 模板窗口大小
            search_window: 搜索窗口大小
        """
        return cv2.fastNlMeansDenoising(img, None, h, template_window, search_window)
    
    def gaussian_denoise(self, img: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        高斯濾波去噪 (傳統方法)
        
        Args:
            img: 輸入圖像
            kernel_size: 核大小 (必須為奇數)
            sigma: 高斯核標準差
        """
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    def bilateral_denoise(self, img: np.ndarray, d: int = 9, 
                         sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        雙邊濾波去噪
        保留邊緣的同時去噪
        
        Args:
            img: 輸入圖像
            d: 濾波器直徑
            sigma_color: 顏色空間標準差
            sigma_space: 坐標空間標準差
        """
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    def multi_scale_denoise(self, img: np.ndarray) -> np.ndarray:
        """
        多尺度去噪 (論文推薦方法)
        結合不同尺度的去噪效果
        """
        # 小尺度：保留細節
        small_scale = cv2.GaussianBlur(img, (3, 3), 0.5)
        # 中尺度：平衡
        mid_scale = cv2.GaussianBlur(img, (5, 5), 1.0)
        # 大尺度：去除粗糙噪聲
        large_scale = cv2.GaussianBlur(img, (7, 7), 1.5)
        
        # 加權融合
        result = (0.5 * small_scale + 0.3 * mid_scale + 0.2 * large_scale).astype(np.uint8)
        return result
    
    def clahe_enhance(self, img: np.ndarray, clip_limit: float = 2.0, 
                     tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        CLAHE對比度增強
        
        Args:
            img: 輸入圖像
            clip_limit: 對比度限制閾值
            tile_grid_size: 網格大小
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    
    def adaptive_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """
        自適應直方圖均衡化
        """
        return exposure.equalize_adapthist(img, clip_limit=0.03)
    
    def morphological_enhancement(self, img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        形態學增強 (2024最新方法)
        使用多尺度形態學操作增強筆劃
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 閉運算：連接斷裂的筆劃
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # 開運算：去除小噪點
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def skeletonize_strokes(self, img: np.ndarray, method: str = 'zhang') -> np.ndarray:
        """
        骨架化處理：提取筆劃中心線
        
        Args:
            img: 輸入圖像 (需為二值圖)
            method: 'zhang' 或 'lee'
        """
        # 確保是二值圖
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        binary = binary.astype(bool)
        
        # 骨架化
        if method == 'zhang':
            skeleton = skeletonize(binary, method='zhang')
        else:
            skeleton = skeletonize(binary, method='lee')
        
        return (skeleton * 255).astype(np.uint8)
    
    def extract_stroke_width(self, img: np.ndarray) -> np.ndarray:
        """
        提取筆劃寬度信息 (保留粗細變化特徵)
        用於分析PD患者的筆劃不穩定性
        """
        # 距離變換
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 歸一化到0-255
        dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        return dist_transform.astype(np.uint8)
    
    def resize_with_padding(self, img: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """
        調整大小並保持長寬比
        
        Args:
            img: 輸入圖像
            maintain_aspect: 是否保持長寬比
        """
        if not maintain_aspect:
            return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        h, w = img.shape[:2]
        target_h, target_w = self.target_size
        
        # 計算縮放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 調整大小
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 添加填充
        result = np.full((target_h, target_w), 255, dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result
    
    def get_augmentation_pipeline(self, stroke_aware: bool = True) -> A.Compose:
        """
        獲取數據增強管道 (2023最新筆劃感知增強)
        
        Args:
            stroke_aware: 是否使用筆劃感知增強
        """
        if stroke_aware:
            # 筆劃感知增強：參數更保守，避免破壞筆劃特徵
            transform = A.Compose([
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.1, 
                    rotate_limit=5, 
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255
                ),
                A.ElasticTransform(
                    alpha=50,  # 較小的變形強度
                    sigma=5,   # 保持筆劃連續性
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255
                ),
                A.GridDistortion(
                    num_steps=5, 
                    distort_limit=0.1, 
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255
                ),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            ])
        else:
            # 標準增強
            transform = A.Compose([
                A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.2, 
                    rotate_limit=15, 
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255
                ),
                A.ElasticTransform(alpha=100, sigma=10, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])
        
        return transform
    
    def apply_augmentation(self, img: np.ndarray, num_augmented: int = 5) -> List[np.ndarray]:
        """
        應用數據增強
        
        Args:
            img: 輸入圖像
            num_augmented: 生成的增強樣本數量
        """
        transform = self.get_augmentation_pipeline(stroke_aware=True)
        augmented_images = []
        
        for _ in range(num_augmented):
            augmented = transform(image=img)['image']
            augmented_images.append(augmented)
        
        return augmented_images
    
    def full_preprocessing_pipeline(self, img: np.ndarray, 
                                   denoise_method: str = 'nlm',
                                   enhance: bool = True,
                                   skeletonize: bool = False) -> dict:
        """
        完整預處理管道
        
        Args:
            img: 輸入圖像
            denoise_method: 'nlm', 'gaussian', 'bilateral', 'multi_scale'
            enhance: 是否進行對比度增強
            skeletonize: 是否進行骨架化
            
        Returns:
            包含各階段處理結果的字典
        """
        results = {'original': img}
        
        # 1. 去噪
        if denoise_method == 'nlm':
            denoised = self.non_local_means_denoise(img)
        elif denoise_method == 'gaussian':
            denoised = self.gaussian_denoise(img)
        elif denoise_method == 'bilateral':
            denoised = self.bilateral_denoise(img)
        elif denoise_method == 'multi_scale':
            denoised = self.multi_scale_denoise(img)
        else:
            denoised = img
        results['denoised'] = denoised
        
        # 2. 對比度增強
        if enhance:
            enhanced = self.clahe_enhance(denoised)
            enhanced = self.morphological_enhancement(enhanced)
            results['enhanced'] = enhanced
        else:
            enhanced = denoised
            results['enhanced'] = enhanced
        
        # 3. 調整大小
        resized = self.resize_with_padding(enhanced, maintain_aspect=True)
        results['resized'] = resized
        
        # 4. 骨架化 (可選)
        if skeletonize:
            skeleton = self.skeletonize_strokes(resized)
            results['skeleton'] = skeleton
        
        # 5. 提取筆劃寬度信息
        stroke_width = self.extract_stroke_width(resized)
        results['stroke_width'] = stroke_width
        
        return results
    
    def visualize_pipeline(self, results: dict, save_path: str = None):
        """
        可視化預處理流程
        
        Args:
            results: full_preprocessing_pipeline返回的結果
            save_path: 保存路徑 (可選)
        """
        n_images = len(results)
        fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
        
        if n_images == 1:
            axes = [axes]
        
        for ax, (title, img) in zip(axes, results.items()):
            ax.imshow(img, cmap='gray')
            ax.set_title(title.replace('_', ' ').title())
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ===== 使用範例 =====
def example_usage():
    """使用範例"""
    
    # 初始化預處理器
    preprocessor = PDHandwritingPreprocessor(target_size=(224, 224))
    
    # 載入圖像
    img = preprocessor.load_image('path_to_your_image.jpg')
    
    # 方法1: 完整管道處理
    print("執行完整預處理管道...")
    results = preprocessor.full_preprocessing_pipeline(
        img, 
        denoise_method='multi_scale',  # 推薦使用多尺度去噪
        enhance=True,
        skeletonize=True
    )
    
    # 可視化結果
    preprocessor.visualize_pipeline(results, save_path='preprocessing_result.png')
    
    # 方法2: 單獨測試不同去噪方法
    print("\n比較不同去噪方法...")
    comparison = {
        'Original': img,
        'NLM': preprocessor.non_local_means_denoise(img),
        'Gaussian': preprocessor.gaussian_denoise(img),
        'Bilateral': preprocessor.bilateral_denoise(img),
        'Multi-Scale': preprocessor.multi_scale_denoise(img)
    }
    preprocessor.visualize_pipeline(comparison, save_path='denoise_comparison.png')
    
    # 方法3: 數據增強
    print("\n生成數據增強樣本...")
    augmented_samples = preprocessor.apply_augmentation(
        results['resized'], 
        num_augmented=5
    )
    
    # 可視化增強樣本
    aug_dict = {f'Aug_{i+1}': aug for i, aug in enumerate(augmented_samples)}
    preprocessor.visualize_pipeline(aug_dict, save_path='augmentation_samples.png')
    
    print("\n預處理完成！")
    return results, augmented_samples


if __name__ == "__main__":
    # 執行範例
    # results, augmented = example_usage()
    
    print("PD手寫數據預處理工具包已就緒！")
    print("\n推薦使用流程：")
    print("1. 多尺度去噪 (multi_scale) - 最佳效果")
    print("2. CLAHE + 形態學增強")
    print("3. 保持長寬比調整大小")
    print("4. 骨架化 (用於分析運動軌跡)")
    print("5. 筆劃感知數據增強")