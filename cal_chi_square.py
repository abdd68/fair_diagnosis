import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os
import re

class Metadata:
    def __init__(self, root):
        self.root = root

    def celeba(self, sensitive_attr, target_attr):
        file_path = os.path.join(self.root, 'list_attr_celeba.txt')
        df = pd.read_csv(file_path, sep='\s+', skiprows=1)
        contingency_table = pd.crosstab(df[sensitive_attr], df[target_attr])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p
    
    def chexpert(self, sensitive_attr, target_attr):
        file_path = os.path.join(self.root, 'chexpert.csv')
        df = pd.read_csv(file_path)
        contingency_table = pd.crosstab(df[sensitive_attr], df[target_attr])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p

    def mimic_cxr(self, sensitive_attr, target_attr):
        file_path = os.path.join(self.root, 'mimic-cxr.csv')
        df = pd.read_csv(file_path)
        contingency_table = pd.crosstab(df[sensitive_attr], df[target_attr])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p

    def TCGA(self, sensitive_attr, target_attr):
        file_path = os.path.join(self.root, 'tuad_1.txt')
        df = pd.read_csv(file_path, sep='\t')

        def classify_stage(stage):
            import re
            match = re.search(r'Stage (\w+)', stage)
            if match:
          
                
                roman_numeral = re.match(r'[IVX]+', match.group(1))
                
                if roman_numeral:
                    roman_numeral = roman_numeral.group(0)
                    
                    if roman_numeral in ['I', 'II']:
                        return 'Early'
                    elif roman_numeral in ['III', 'IV']:
                        return 'Late'
            return 'Unknown'

        df['Stage_Class'] = df['ajcc_pathologic_tumor_stage'].apply(classify_stage)

    
        df_filtered = df[df['Stage_Class'] != 'Unknown']

       
        contingency_table = pd.crosstab(df_filtered[sensitive_attr], df_filtered['Stage_Class'])
        

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p

if __name__ == "__main__":
    dataset = 'TCGA'
    sensitive_attr = 'gender'
    # if dataset == 'TCGA' target_attr = any
    target_attr = 'Stage_Class'
    metadata = Metadata('data')
    chi2, p = getattr(metadata, dataset)(sensitive_attr, target_attr)
    print(f"Chi-square statistic: {chi2}")
    print(f"P-value: {p}")
