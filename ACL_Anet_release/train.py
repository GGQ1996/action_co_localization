import os
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from logger import Logger
from model import BackboneNet
from dataset import SingleVideoDataset
from utils import get_dataset, load_config_file
from utils import get_label_2_video, get_sample_list
from utils import get_single_label_dict

device = torch.device('cuda')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config-file', type=str)
    parser.add_argument('--train-subset-name', type=str)
    parser.add_argument('--test-subset-name', type=str)

    parser.add_argument('--test-log', dest='test_log', action='store_true')
    parser.add_argument('--no-test-log', dest='test_log', action='store_false')
    parser.set_defaults(test_log=True)

    args = parser.parse_args()

    print(args.config_file)
    print(args.train_subset_name)
    print(args.test_subset_name)
    print(args.test_log)

    all_params = load_config_file(args.config_file)
    locals().update(all_params)


    def test(model, loader, modality):

        assert (modality in ['both', 'rgb', 'flow'])

        pred_score_dict = {}
        label_dict = {}

        correct = 0
        total_cnt = 0
        total_loss = {
            'cls': 0,
            'div': 0,
            'norm': 0,
            'sum': 0,
        }

        criterion = nn.CrossEntropyLoss(reduction='elementwise_mean')

        with torch.no_grad():

            model.eval()

            for _, data in enumerate(loader):  # No shuffle

                video_name = data['video_name'][0]
                label = data['label'].to(device)
                weight = data['weight'].to(device).float()

                if label.item() == action_class_num:
                    continue
                else:
                    total_cnt += 1

                if modality == 'both':
                    rgb = data['rgb'].to(device).squeeze(0)
                    flow = data['flow'].to(device).squeeze(0)
                    model_input = torch.cat([rgb, flow], dim=2)
                elif modality == 'rgb':
                    model_input = data['rgb'].to(device).squeeze(0)
                else:
                    model_input = data['flow'].to(device).squeeze(0)

                model_input = model_input.transpose(2, 1)
                _, _, out, scores, _ = model(model_input)

                out = out.mean(0, keepdim=True)

                loss_cls = criterion(out, label) * weight
                total_loss['cls'] += loss_cls.item()

                if diversity_reg:
                    loss_div = get_diversity_loss(scores) * weight
                    loss_div = loss_div * diversity_weight

                    loss_norm = get_norm_regularization(scores) * weight
                    loss_norm = loss_norm * diversity_weight

                    total_loss['div'] += loss_div.item()
                    total_loss['norm'] += loss_norm.item()

                out = out[:, :action_class_num]  # Remove bg
                pred = torch.argmax(out, dim=1)
                correct += (pred.item() == label.item())

                ###############

                video_key = ''.join(video_name.split('-')
                                    [:-1])  # remove content after the last -

                pred_score_dict[video_key] = out.cpu().numpy()

                if video_key not in label_dict.keys():
                    label_dict[video_key] = np.zeros((1, action_class_num))

                label_dict[video_key][0, label.item()] = 1
                ###############

        accuracy = correct / total_cnt
        total_loss[
            'sum'] = total_loss['cls'] + total_loss['div'] + total_loss['norm']
        avg_loss = {k: v / total_cnt for k, v in total_loss.items()}

        ##############
        pred_score_matrix = []
        label_matrix = []
        for k, v in pred_score_dict.items():
            pred_score_matrix.append(v)
            label_matrix.append(label_dict[k])

        # don't use
        mean_ap = 0

        # _, mean_ap = eval_thumos_recog(
        #     np.concatenate(pred_score_matrix, axis=0),
        #     np.concatenate(label_matrix, axis=0), action_class_num)

        return accuracy, avg_loss, mean_ap


    def my_train(train_train_loader, train_test_loader, test_test_loader, modality,
                 naming, label_2_video, num_of_video):

        assert (modality in ['both', 'rgb', 'flow'])

        log_dir = os.path.join('logs', naming, modality)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = Logger(log_dir)

        save_dir = os.path.join('models', naming)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if modality == 'both':
            model = BackboneNet(in_features=feature_dim * 2,
                                **model_params).to(device)
        else:
            model = BackboneNet(in_features=feature_dim,
                                **model_params).to(device)

        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)

        if learning_rate_decay:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[max_step_num // 2], gamma=0.1)

        optimizer.zero_grad()

        criterion = nn.CrossEntropyLoss(reduction='mean')

        update_step_idx = 0
        single_video_idx = 0
        loss_recorder = {
            'cls': 0,
            'div': 0,
            'norm': 0,
            'sum': 0,
            'sep': 0,
            'trip': 0
        }

        # add
        all_class = list(label_2_video.keys())
        all_class.sort()

        # add state sample video

        while update_step_idx < max_step_num:

            idx, label_idx = get_sample_list(num_of_video, label_2_video, batch_size, n_similar)

            print("sample index")
            print(idx)
            print("label index")
            print(label_idx)
            atten_set = []
            fea_set = []
            global_atten_set = []
            base_fea_set = []

            loss = torch.Tensor([0]).to(device)
            for cnt in range(len(idx)):
                tmp_id = idx[cnt]
                tmp_label = None
                if cnt < len(label_idx):
                    tmp_label = label_idx[cnt]

                data = train_train_loader[tmp_id]

                model.train()

                single_video_idx += 1

                label = torch.from_numpy(data['label']).long().to(device)
                weight = torch.from_numpy(data['weight']).float().to(device)

                if modality == 'both':
                    rgb = torch.from_numpy(data['rgb']).float().to(device)
                    flow = torch.from_numpy(data['flow']).float().to(device)
                    if len(rgb.shape) == 2:
                        rgb = rgb.unsqueeze(0)
                    if len(flow.shape) == 2:
                        flow = flow.unsqueeze(0)

                    model_input = torch.cat([rgb, flow], dim=2)
                elif modality == 'rgb':
                    model_input = torch.from_numpy(data['rgb']).float().to(device)
                else:
                    model_input = torch.from_numpy(data['flow']).float().to(device)

                # print(model_input.shape)
                if len(model_input.shape) == 2:
                    model_input = model_input.unsqueeze(0)
                if len(label.shape) == 0:
                    label = label.unsqueeze(0)
                    weight = weight.unsqueeze(0)

                # print(model_input.shape)
                model_input = model_input.transpose(2, 1)
                avg_score, att_weight, out, scores, feature_dict = model(model_input)

                # add
                if tmp_label is not None:
                    atten_set.append(avg_score[:, :, tmp_label:tmp_label + 1])
                    fea_set.append(feature_dict['fuse_feature'])
                    global_atten_set.append(att_weight)

                    base_fea_set.append(feature_dict['base_feature'])

                loss_cls = criterion(out, label) * weight

                # add sep flag
                sep_flag = (single_video_idx % batch_size == 0)

                if sep_flag:
                    sep_loss_weight = 1
                    trip_loss_weight = 0

                    loss = loss + loss_cls
                    loss_recorder['cls'] += loss_cls.item()

                    if len(global_atten_set) > 0:
                        loss_sep = sep_loss_weight * sep_loss(atten_set, fea_set, device)

                        # add separation loss and cluster loss
                        loss_trip = trip_loss_weight * triplet_loss(global_atten_set, base_fea_set, label_idx, device)
                        loss = loss + loss_sep + loss_trip

                        loss_recorder['sep'] = loss_sep.item()
                        loss_recorder['trip'] = loss_trip.item()

                    loss.backward()

                else:
                    loss = loss + loss_cls
                    loss_recorder['cls'] += loss_cls.item()

                # loss is the cumulative sum
                loss_recorder['sum'] = loss.item()

                # Test and Update
                if single_video_idx % batch_size == 0:
                    # calculate sep loss

                    # Test
                    if update_step_idx % log_freq == 0:
                        pass

                    # Batch Update
                    update_step_idx += 1

                    for k, v in loss_recorder.items():
                        logger.scalar_summary('Loss_{}_ps'.format(k),
                                              v / batch_size, update_step_idx)

                        loss_recorder[k] = 0

                    optimizer.step()
                    optimizer.zero_grad()

                    if learning_rate_decay:
                        scheduler.step()

                    if update_step_idx in check_points:
                        torch.save(
                            model.state_dict(),
                            os.path.join(
                                save_dir,
                                'model-{}-{}'.format(modality,
                                                     update_step_idx)))

                    if update_step_idx >= max_step_num:
                        break


    train_dataset_dict = get_dataset(dataset_name=dataset_name,
                                     subset=args.train_subset_name,
                                     file_paths=file_paths,
                                     sample_rate=sample_rate,
                                     base_sample_rate=base_sample_rate,
                                     action_class_num=action_class_num,
                                     modality='both',
                                     feature_type=feature_type,
                                     feature_oversample=feature_oversample,
                                     temporal_aug=True,
                                     load_background=with_bg)

    train_train_dataset = SingleVideoDataset(
        train_dataset_dict,
        single_label=True,
        random_select=True,
        max_len=training_max_len)  # To be checked

    train_test_dataset = SingleVideoDataset(train_dataset_dict,
                                            single_label=True,
                                            random_select=False,
                                            max_len=None)

    train_train_loader = torch.utils.data.DataLoader(train_train_dataset,
                                                     batch_size=1,
                                                     pin_memory=True,
                                                     shuffle=True)

    train_test_loader = torch.utils.data.DataLoader(train_test_dataset,
                                                    batch_size=1,
                                                    pin_memory=True,
                                                    shuffle=False)

    if args.test_log:

        test_dataset_dict = get_dataset(dataset_name=dataset_name,
                                        subset=args.test_subset_name,
                                        file_paths=file_paths,
                                        sample_rate=sample_rate,
                                        base_sample_rate=base_sample_rate,
                                        action_class_num=action_class_num,
                                        modality='both',
                                        feature_type=feature_type,
                                        feature_oversample=feature_oversample,
                                        temporal_aug=True,
                                        load_background=False)

        test_test_dataset = SingleVideoDataset(test_dataset_dict,
                                               single_label=True,
                                               random_select=False,
                                               max_len=None)

        test_test_loader = torch.utils.data.DataLoader(test_test_dataset,
                                                       batch_size=1,
                                                       pin_memory=True,
                                                       shuffle=False)
    else:

        test_test_loader = None

    single_label_train_dict = get_single_label_dict(train_dataset_dict)
    label_2_video = get_label_2_video(single_label_train_dict)
    num_of_video = len(single_label_train_dict)

    for run_idx in range(train_run_num):
        naming = '{}-run-{}'.format(experiment_naming, run_idx)
        my_train(train_train_dataset, train_test_loader, test_test_loader, 'rgb',
                 naming, label_2_video, num_of_video)
