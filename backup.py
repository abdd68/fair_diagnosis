def process_dataset():
    if args.dataset == 'mimic-cxr':
        mimic_cxr_path = "../datasets/mimic-cxr/mimic-cxr-jpg-small/2.0.0"  # 修改为实际路径
        chexpert_csv_path = os.path.join(mimic_cxr_path, "mimic-cxr-2.0.0-chexpert.csv")
        patients_csv_path = os.path.join(mimic_cxr_path, "patients.csv")
        metadata_csv_path = os.path.join(mimic_cxr_path,'mimic-cxr-2.0.0-metadata.csv')
        
        # 读取 CSV 文件
        chexpert_df = pd.read_csv(chexpert_csv_path)
        patients_df = pd.read_csv(patients_csv_path)
        mimic_cxr_metadata = pd.read_csv(metadata_csv_path)
        
        # 只选择需要的列，并确保包含 ViewPosition 信息
        chexpert_df = chexpert_df[["study_id", "subject_id", "Pneumonia"]]
        patients_df = patients_df[["subject_id", "gender"]]
        merged_df = pd.merge(chexpert_df, patients_df, on="subject_id")
        # 加入 ViewPosition 列，合并 chexpert_df 和 mimic-cxr-2.0.0-metadata.csv
        pa_df = mimic_cxr_metadata[mimic_cxr_metadata["ViewPosition"] == "PA"]
        # pa_df = mimic_cxr_metadata
        # 如果需要合并，可以在过滤后的 chexpert_df 上进行 merge
        mimic_metadata = pd.merge(pa_df, merged_df, on="study_id")

        mimic_metadata["subject_id"] = mimic_metadata["subject_id_x"]  # 或者使用 subject_id_y
        mimic_metadata = mimic_metadata.drop(columns=["subject_id_x", "subject_id_y"])
        
        # 将性别转为 0（男性）和 1（女性）
        mimic_metadata['gender'] = mimic_metadata['gender'].map({'M': 0, 'F': 1})
        
        if args.debug:
            train_metadata, test_metadata = train_test_split(mimic_metadata, test_size=0.01, train_size = 0.09, random_state=args.seed)
        else:
            train_metadata, test_metadata = train_test_split(mimic_metadata, test_size=0.1, train_size = 0.9, random_state=args.seed)
        train_dataset = MimicCXRDataset(metadata=train_metadata, root_dir=mimic_cxr_path, transform=transform)
        test_dataset = MimicCXRDataset(metadata=test_metadata, root_dir=mimic_cxr_path, transform=transform)
    elif args.dataset == 'chexpert':
        root_dir = '../datasets'
        train_dataset = CheXpertDataset(csv_file=os.path.join(root_dir, 'CheXpert-v1.0-small/train.csv'), 
                                root_dir=root_dir, 
                                transform=transform, debug=args.debug)
        test_dataset = CheXpertDataset(csv_file=os.path.join(root_dir, 'CheXpert-v1.0-small/valid.csv'), 
                               root_dir=root_dir, 
                               transform=transform, debug=False)
        
    elif args.dataset == 'tcga':
        root_dir = '../datasets/TCGA-LUAD/manifest-1Rd7jPNd5199284876140322680'
        train_dataset = TcgaDataset(csv_file=os.path.join(root_dir, 'metadata.csv'), root_dir=root_dir, transform=transform, debug=args.debug, training = True)
        test_dataset = TcgaDataset(csv_file=os.path.join(root_dir, 'metadata.csv'), root_dir=root_dir, transform=transform, debug=args.debug, training = False)
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32)

    logger.info(f"Number of samples in training dataset: {len(train_dataset)} | testing dataset:{len(test_dataset)}")
    
    return train_dataset, test_dataset, train_loader, test_loader