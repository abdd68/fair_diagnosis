from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader


class MimicCXRDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None):
        # self.metadata = metadata.dropna(subset=["study_id", "subject_id", "Pneumonia", "gender"]).reset_index(drop=True)
        self.metadata = metadata
        self.metadata["Pneumonia"] = self.metadata["Pneumonia"].fillna(0)
        self.metadata["Pneumonia"] = self.metadata["Pneumonia"].replace(-1, 0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 获取影像文件路径
        study_id = int(self.metadata.iloc[idx]["study_id"])
        subject_id = int(self.metadata.iloc[idx]["subject_id"])

        # 构造图像文件路径，例如：root_dir/p10/p10000032/s50414267/*.jpg
        subject_folder = f"p{str(subject_id)[:2]}"
        study_folder = f"p{subject_id}/s{study_id}"
        image_path = os.path.join(self.root_dir, "files", subject_folder, study_folder)

        # 加载图像（假设图像文件夹下有 jpg 文件）
        image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
        image = Image.open(os.path.join(image_path, image_files[0])).convert('RGB')  # 加载第一张图像

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 提取肺炎标签和性别信息
        pneumonia_label = int(self.metadata.iloc[idx]["Pneumonia"])  # 转换为整数类型
        gender_label = np.float32(self.metadata.iloc[idx]["gender"])  # 转换为浮点类型

        return image, pneumonia_label, gender_label

    # 新增：计算正样本数除以负样本数
    def calculate_neg_pos_ratio(self):
        # 正样本：Pneumonia 标签为 1 的样本
        positive_count = (self.metadata["Pneumonia"] == 1).sum()
        # 负样本：Pneumonia 标签为 0 的样本
        negative_count = (self.metadata["Pneumonia"] == 0).sum()

        # 防止负样本数为 0 导致除以 0 错误
        if negative_count == 0:
            return float('inf')  # 如果没有负样本，返回无穷大

        # 计算正负样本比
        neg_pos_ratio = float(negative_count / positive_count)
        return neg_pos_ratio
    
    
class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, debug=False):
        # 读取CSV文件
        self.data_frame = pd.read_csv(csv_file)
        # self.data_frame = self.data_frame.dropna(subset=["Atelectasis"]).reset_index(drop=True)
        self.data_frame["Atelectasis"] = self.data_frame["Pneumonia"].fillna(0)
        self.data_frame["Atelectasis"] = self.data_frame["Atelectasis"].replace(-1, 0)
        # 如果是调试模式，仅取前10%的数据
        if debug:
            self.data_frame = self.data_frame.sample(frac=0.1, random_state=42).reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取图像路径
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        
        # 获取性别信息并转换为数字
        gender_label = self.data_frame.iloc[idx, 1]
        gender_label = np.float32(1.0) if gender_label == 'Male' else np.float32(0.0)  # 男性为1，女性为0

        # 获取Pneumonia的标签
        target_label = self.data_frame.iloc[idx, 13]
        target_label = int(target_label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, target_label, gender_label
    
    def calculate_neg_pos_ratio(self):
        # 计算Pneumonia（target_label）的正负样本比
        positive_count = (self.data_frame["Atelectasis"] == 1).sum()
        negative_count = (self.data_frame["Atelectasis"] == 0).sum()

        # 防止负样本数为0导致除以0错误
        if positive_count == 0:
            return float('inf')  # 如果没有正样本，返回无穷大

        # 计算负正样本比
        neg_pos_ratio = float(negative_count / positive_count)
        return neg_pos_ratio
    
from sklearn.model_selection import train_test_split
import pydicom
class TcgaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, debug=False, training = False):
        # 读取CSV文件
        # 1. 读取 metadata.csv
        self.training = training
        metadata_path = csv_file  # 请替换为实际的 metadata.csv 路径
        metadata = pd.read_csv(metadata_path)

        # 2. 读取临床数据文件
        clinical_data_path = '../datasets/TCGA-LUAD/gdc_download_clinical_luad/nationwidechildrens.org_clinical_patient_luad.txt'
        clinical_data = pd.read_csv(clinical_data_path, sep="\t")

        # 3. 筛选出临床数据中包含 bcr_patient_barcode 和 ajcc_pathologic_tumor_stage 列的数据
        clinical_data_reduced = clinical_data[['gender','bcr_patient_barcode', 'ajcc_pathologic_tumor_stage']]
        
        # 4. 合并 metadata 和 clinical_data_reduced，使用 bcr_patient_barcode 作为连接键
        merged_data = pd.merge(metadata, clinical_data_reduced, left_on="Subject ID", right_on="bcr_patient_barcode", how="left")
        merged_data = merged_data.dropna(subset=['bcr_patient_barcode', 'ajcc_pathologic_tumor_stage'])
        
        stages_to_zero = ['Stage IA', 'Stage IB', 'Stage IIA', 'Stage IIB', '[Discrepancy]']
        merged_data['ajcc_pathologic_tumor_stage'] = merged_data['ajcc_pathologic_tumor_stage'].apply(
            lambda x: 0 if x in stages_to_zero else 1)
        
        # 假设 merged_data 已经加载
        expanded_data = []

        # 遍历每一行，按 `Number of Images` 将图片集拆分成单独的行
        for idx, row in merged_data.iterrows():
            num_images = int(row["Number of Images"])
            base_location = row["File Location"].replace("\\", "/")

            # 检查 File Location 是否是一个有效的目录
            list_dir = os.path.join(root_dir,base_location)
            if os.path.isdir(list_dir):
                image_files = os.listdir(list_dir)
                # 按 `Number of Images` 限制图片数量
                image_files = image_files[:num_images]
                
                # 为每个图像创建一个新行
                for img_file in image_files:
                    new_row = row.copy()
                    # 更新 File Location 为具体图片的路径
                    new_row["File Location"] = os.path.join(base_location, img_file)
                    expanded_data.append(new_row)
                

        # 构建扩展后的 DataFrame
        merged_data = pd.DataFrame(expanded_data)
        
        
        if debug:
            train_metadata, test_metadata = train_test_split(merged_data, test_size=0.01, train_size = 0.09, random_state=42)
        else:
            train_metadata, test_metadata = train_test_split(merged_data, test_size=0.1, train_size = 0.9, random_state=42)
            
        if training:
            self.data_frame = train_metadata
        else:
            self.data_frame = test_metadata
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 获取行数据
        row = self.data_frame.iloc[idx]
        
        # 获取图像路径
        img_path = os.path.join(self.root_dir, row["File Location"])
        
        # 加载图像
        dicom = pydicom.dcmread(img_path)
        img_array = dicom.pixel_array
        image = Image.fromarray(img_array).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 提取标签和元数据
        ajcc_stage = int(row["ajcc_pathologic_tumor_stage"])  # 假设是整数标签
        gender = row["gender"]  # 示例：性别字段
        gender_label = np.float32(1.0) if gender == 'MALE' else np.float32(0.0)
        # 你可以根据需要提取更多的元数据字段
        
        return image, ajcc_stage, gender_label  # 返回图像、标签和其他元数据
    
    
    def calculate_neg_pos_ratio(self):
        # 计算Pneumonia（target_label）的正负样本比
        positive_count = (self.data_frame["ajcc_pathologic_tumor_stage"] == 1).sum()
        negative_count = (self.data_frame["ajcc_pathologic_tumor_stage"] == 0).sum()

        # 防止负样本数为0导致除以0错误
        if positive_count == 0:
            return float('inf')  # 如果没有正样本，返回无穷大

        # 计算负正样本比
        neg_pos_ratio = float(negative_count / positive_count)
        return neg_pos_ratio

class NIHCXRDataset(Dataset):
    def __init__(self, image_paths, data_labels, gender_labels, root_dir, transforms=None):
        self.image_paths = image_paths
        self.data_labels = data_labels
        self.gender_labels = gender_labels
        self.root_dir = root_dir
        self.transforms = transforms
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, 'imgs',self.image_paths[idx])).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, self.data_labels[idx], self.gender_labels[idx]
        
class NIHCXR():
    def __init__(self, args, normalize=False):
        self.args = args
        
        self.norm_layer = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        )
        
        self.transforms = [
            transforms.CenterCrop(178),
			transforms.Resize((224, 224)),
			transforms.ToTensor()
        ]
        
        if normalize:
            self.transforms.append(self.norm_layer)
            
        self.transforms = transforms.Compose(self.transforms)
        
    def _extract_data(self, data_list):
        '''
        data_labels: n * 15, 15 个标签，每个标签为 0 或 1, print(mlb.classes_)
        gender_labels: n * 1, 0 = female, 1 = male
        
        '''
        with open(os.path.join(self.args.data_dir, data_list), 'r') as file:
            image_index_list = file.read().splitlines()

        df = pd.read_csv(os.path.join(self.args.data_dir, 'Data_Entry_2017.csv'))
        df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
        df['Patient Gender'] = df['Patient Gender'].apply(lambda x: 1 if x == 'M' else 0)
        
        df = df[df['Image Index'].isin(image_index_list)]

        mlb = MultiLabelBinarizer()
        image_paths = df['Image Index'].values
        data_labels = mlb.fit_transform(df['Finding Labels'])
        gender_labels = df['Patient Gender'].values

        return image_paths, data_labels, gender_labels
    
    def data_loaders(self, **kwargs):
        train_image_paths, train_data_labels, train_gender_labels = self._extract_data("train_val_list.txt")
        test_image_paths, test_data_labels, test_gender_labels = self._extract_data("test_list.txt")
        
        
        train_dataset = NIHCXRDataset(train_image_paths, train_data_labels, train_gender_labels, self.args.data_dir, self.transforms)
        test_dataset = NIHCXRDataset(test_image_paths, test_data_labels, test_gender_labels, self.args.data_dir, self.transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        return train_loader, test_loader
        
if __name__ == "__main__":
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument("--data_dir", type=str, default="nihdata")
        parser.add_argument("--batch_size", type=int, default=4)
        return parser.parse_args()
    
    args = parse_args()
    nih = NIHCXR(args)
    train_loader, test_loader = nih.data_loaders()
    img, label, gender = next(iter(train_loader))
    print(img.shape)
    print(label)
    print(gender)
