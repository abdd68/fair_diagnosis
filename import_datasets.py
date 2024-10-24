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
