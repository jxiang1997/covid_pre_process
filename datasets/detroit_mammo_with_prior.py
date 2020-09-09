from datasets.abstract_mammo_with_prior import Abstract_Mammo_Cancer_With_Prior_Dataset
from torch.utils import data
from tqdm import tqdm

import json
import numpy as np
import pdb
import os

SUMMARY_MSG = "Contructed Detroit Mammo with prior dataset with {} records, {} exams, {} patients, and the following class balance \n {}"

METADATA_FILENAME = "detroit_metadata_reformatted.json"

class Detroit_Mammo_Cancer_With_Prior_Dataset(Abstract_Mammo_Cancer_With_Prior_Dataset):
    """
    Dataset object for detroit mammogram data. Dataset object is associated with the specified metadata file and also has a create_dataset method and a task. 
    On a forward pass, this dataset object will return an mammogram image concatenated with a prior of the same view.
    """

    def create_dataset(self, split_group: str, img_dir: str):
        """
            Creates a pytorch dataset object from a metadata file (self.metadata_json)

            params: split_group - ['train'|'dev'|'test'].
            img_dir: path to directory containing mammogram images

        """
        dataset = []
        for mrn_row in tqdm(self.metadata_json):
            ssn, split, exams = mrn_row['patient_id'], mrn_row['split'], mrn_row['accessions']
            if not split == split_group:
                continue

            year_to_exam = {}
            for exam in exams:
                year = exam['date']

            left_ccs, left_mlos, right_ccs, right_mlos = self.get_image_paths_by_views(exam)

            # Create dictionary of all exams for every patient
            year_to_exam[year] = {
                'L CC': left_ccs,
                'L MLO': left_mlos,
                'R CC': right_ccs,
                'R MLO': right_mlos,
                'ssn': ssn,
                'exam': exam
            }

        all_views = ['L CC', 'L MLO', 'R CC', 'R MLO']
        all_years = list(reversed(sorted(year_to_exam.keys())))

        for index, year in enumerate(all_years):

            exam = year_to_exam[year]
            prior_exams = [ (prior_year, year_to_exam[prior_year]) for prior_year in all_years[index+1:]]

            for view in all_views:
                if len(exam[view]) == 0:
                    continue

                prior_with_view = [ (prior_year, prior) for prior_year, prior in prior_exams if len(prior[view]) > 0]

                for prior_year, prior in prior_with_view:

                    dataset.append({
                        'paths': [exam[view][0], prior[view][0]],
                        'exam': exam['exam'],
                        'prior_exam': prior['exam'],
                        'year': year,
                        'prior_year': prior_year,
                        'ssn':ssn,
                        'time_difference': year - prior_year

                    })
        
        return dataset

    @property
    def METADATA_FILENAME(self):
        return METADATA_FILENAME

    @staticmethod
    def set_args(args):
        args.num_classes = 2
        args.multi_image = True
        args.num_images = 2


    