import gspread
import argparse
import os
import string
import math
import time

from oauth2client.service_account import ServiceAccountCredentials


_CANT_FIND_PARAMETERS = r"""
Couldn't find the cell that contained "START PARAMETER".
Please add "START PARAMETER" to the front of the parameter column.
"""
_LIMIT_COLUMN_ERROR = r"""
We limit the max column to ZZ.
"""
_NO_VALID_EXPERIMENT_ERROR = r"""
Couldn't find the cell that contained "END EXPERIMENT".
Please add "END EXPERIMENT" to the last row.
"""

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class SetSheet(object):
    def __init__(self, args, sheet_name):
        self.eval_end_point = args.eval_end_point
        self.sheet_name = sheet_name
        self.gpu_number = args.gpu_number


class AutoCommand(SetSheet):
    def __init__(self, arguments, sheet_name):
        super(AutoCommand, self).__init__(arguments, sheet_name)
        # Dataset Setting
        self.end_line_number = 0
        self.param_column = []
        self.sheet_column = []
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive',
        ]
        json_file_name = '개인정보상 삭제' # put a json file name which contains your own key.
        credentials = ServiceAccountCredentials.from_json_keyfile_name(json_file_name, scope)
        self.gc = gspread.authorize(credentials)
        self.spreadsheet_url = "개인정보상 삭제" # put a spreadsheet url. / Caution: gid value should be 0.
        self.worksheet = self.read_sheet()
        self.params_list = self.set_parameter()

    def read_sheet(self):
        doc = self.gc.open_by_url(self.spreadsheet_url)
        return doc.worksheet(self.sheet_name)

    def split_arch(self, s):
        head = s.rstrip('0123456789')
        head = head.split('_')[0]
        return head

    def sheet_column_value_generator(self, column_number):
        all_column_number = column_number + 100
        loop_count = int(math.floor(all_column_number / 26))
        if loop_count >= 24:
            raise NotImplementedError(_LIMIT_COLUMN_ERROR)
        for i in range(loop_count+1):
            front_alphabet = '' if i == 0 else string.ascii_uppercase[i-1]
            last_alphabet_slice_number = 26 if i < loop_count else all_column_number % 26
            self.sheet_column.extend([front_alphabet + j for j in string.ascii_uppercase[:last_alphabet_slice_number]])
        self.sheet_column = self.sheet_column[column_number:]

    def set_parameter(self):
        lines = self.worksheet.get_all_values()
        params_list = []
        param_cnt = 0
        line_cnt = 1
        for line in lines:
            # Line parameter settings
            if line[0] == "START PARAMETER":
                column_number = line.index("END PARAMETER")
                self.param_column = {i: column_name for i, column_name in enumerate(line[1:column_number])}
                self.sheet_column_value_generator(column_number)
                line_cnt += 1
                continue
            # For Write Object Count
            if line[0] == "END EXPERIMENT":
                self.end_line_number = line_cnt
                line_cnt += 1
                break
            # Skip the Null Row
            if line[0] != str(param_cnt):
                line_cnt += 1
                continue

            if len(self.param_column) == 0:
                raise NotImplementedError(_CANT_FIND_PARAMETERS)

            param_dict = {'line_num': line_cnt}

            param_dict.update({self.param_column[i]: line[i+1] for i in range(0, len(self.param_column))})
            params_list.append(param_dict)

            param_cnt += 1
            line_cnt += 1

        return params_list

    def eval_command(self):
        for param_cnt, params in enumerate(self.params_list[self.eval_end_point[0]:self.eval_end_point[1]+1]):

            # finetune code
            command = " deepspeed --include localhost:1,2 /LLaVA/llava/train/train_mem.py " \
                      "--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 " \
                      "--deepspeed /LLaVA/scripts/zero3.json " \
                      "--model_name_or_path liuhaotian/llava-v1.5-7b " \
                      "--version v1 " \
                      "--image_folder '' " \
                      "--vision_tower openai/clip-vit-large-patch14-336 " \
                      "--mm_projector_type mlp2x_gelu " \
                      "--mm_vision_select_layer -2 " \
                      "--mm_use_im_start_end False " \
                      "--mm_use_im_patch_token False " \
                      "--image_aspect_ratio pad " \
                      "--group_by_modality_length True " \
                      "--bf16 True " \
                      "--num_train_epochs 1 " \
                      "--per_device_train_batch_size 32 " \
                      "--per_device_eval_batch_size 4 " \
                      "--gradient_accumulation_steps 1 " \
                      "--evaluation_strategy 'no' " \
                      "--save_strategy 'steps' " \
                      "--save_steps 50000 " \
                      "--save_total_limit 1 " \
                      "--learning_rate 2e-4 " \
                      "--weight_decay 0. " \
                      "--warmup_ratio 0.03 " \
                      "--lr_scheduler_type 'cosine' " \
                      "--logging_steps 1 " \
                      "--tf32 True " \
                      "--model_max_length 2048 " \
                      "--gradient_checkpointing True " \
                      "--dataloader_num_workers 4 " \
                      "--lazy_preprocess True " \
                      "--report_to wandb "


            for i in range(0, len(self.param_column)):
                if params[self.param_column[i]] != '':
                    command += " --{} {} ".format(self.param_column[i], params[self.param_column[i]])

            print(command)
            stream = os.popen(command)
            output_all = stream.readlines()
            output = ["DONE"]

            # Total Score
            sheet_range = '{}:{}'.format(self.sheet_column[1] + str(params["line_num"]),
                                         self.sheet_column[len(output) + 1] + str(params["line_num"]))
            cell_list = self.worksheet.range(sheet_range)
            for idx, val in enumerate(output):
                val = (val.rstrip('\n').split(' ')[-1])
                cell_list[idx].value = val
            self.worksheet.update_cells(cell_list)

    def __call__(self):
        self.eval_command()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_end_point', nargs='+', type=int, default=[0, 100])
    parser.add_argument('--sheet_name', type=str, default='debug')
    parser.add_argument('--gpu_number', type=int, default=0)

    args = parser.parse_args()

    command_module = AutoCommand(args, args.sheet_name)
    command_module()

