import torch
import numpy as np
import warnings
import sklearn.linear_model as lm
import sklearn.metrics as skm

class Evaluator:
    r"""
    A unified evaluation mechanism is provided to ensure that the input data is in the correct format
    and meets the requirements of the evaluation indicators.

    Args:
        eval_metric (str): :obj:`hits@k` refers to the probability of a hit in the first k results.
        K (int): Get from :obj:`hits@k`.
    """

    def __init__(self, eval_metric='hits@50'):

        self.eval_metric = eval_metric
        if 'hits@' in self.eval_metric:
            self.K = int(self.eval_metric.split('@')[1])

    def _parse_and_check_input(self, input_dict: dict):
        r"""Parses and checks whether the type and shape of the input data meets the requirements,
        and the data is converted into a uniform format.

        Args:
            input_dict (dict): The input dictionary.

        """

        if 'hits@' in self.eval_metric:
            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'

            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            return y_pred_pos, y_pred_neg, type_info

        elif 'mrr' == self.eval_metric:

            if not 'y_pred_pos' in input_dict:
                raise RuntimeError('Missing key of y_pred_pos')
            if not 'y_pred_neg' in input_dict:
                raise RuntimeError('Missing key of y_pred_neg')

            y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

            if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
                raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

            if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
                raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

            if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
                if isinstance(y_pred_pos, np.ndarray):
                    y_pred_pos = torch.from_numpy(y_pred_pos)

                if isinstance(y_pred_neg, np.ndarray):
                    y_pred_neg = torch.from_numpy(y_pred_neg)

                y_pred_pos = y_pred_pos.to(y_pred_neg.device)

                type_info = 'torch'
            else:
                type_info = 'numpy'

            if not y_pred_pos.ndim == 1:
                raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

            if not y_pred_neg.ndim == 2:
                raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

            return y_pred_pos, y_pred_neg, type_info

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def eval(self, input_dict: dict):
        r"""Evaluates and return the results(which could be hits or mrr) according to the evaluation metric.

        Args:
            input_dict (dict): The input dictionary.
        """

        if 'hits@' in self.eval_metric:
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_hits(y_pred_pos, y_pred_neg, type_info)
        elif self.eval_metric == 'mrr':
            y_pred_pos, y_pred_neg, type_info = self._parse_and_check_input(input_dict)
            return self._eval_mrr(y_pred_pos, y_pred_neg, type_info)

        else:
            raise ValueError('Undefined eval metric %s' % (self.eval_metric))

    def _eval_hits(self, y_pred_pos, y_pred_neg, type_info):
        r"""Evaluates and returns the value :obj:`hits@k`, which refers to the probability of hitting in the first k results.
        It is to see whether there is a real label in the first k results, if there is a hit success, otherwise it is a hit failure

        Args:
            y_pred_pos (float): The fraction or probability of the sample for which the model predicts a positive sample.
            y_pred_neg (float): The fraction or probability of the sample for which the model predicts a negative example
            type_info (str): The way to address the data.

"""

        if type_info == 'torch':
            res = torch.topk(y_pred_neg, self.K)
            kth_score_in_negative_edges = res[0][:, -1]
            hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

        else:
            kth_score_in_negative_edges = np.sort(y_pred_neg)[-self.K]
            hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

        return {'hits@{}'.format(self.K): hitsK}

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        r"""Evaluates and returns the value :obj:`mrr`: Mean Reciprocal Ranking.

        Args:
            y_pred_pos (float): The fraction or probability of the sample for which the model predicts a positive sample.
            y_pred_neg (float): The fraction or probability of the sample for which the model predicts a negative example
            type_info (str): The way to address the data.
            """

        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            mrr_list = 1. / ranking_list.to(torch.float)
            return mrr_list.mean()
        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg], axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            mrr_list = 1. / ranking_list.astype(np.float32)
            return mrr_list.mean()


def mrr_hit(normal_label: np.ndarray, pos_out: np.ndarray, metric: list = None):
    r"""Evaluates and returns the value of both :obj:`mrr` and :obj:`hits@k`.

    Args:
        normal_label (ndarray): The information of labeling.
        pos_out (ndarray): The information of positive output.
        matric (list): Assigned evaluation index.
        """

    if isinstance(normal_label, np.ndarray) and isinstance(pos_out, np.ndarray):
        pass
    else:
        warnings.warn('it would be better if normal_label and out are all set as np.ndarray')

    results = {}
    if not metric:
        metric = ['mrr', 'hits']

    if 'hits' in metric:
        hits_evaluator = Evaluator(eval_metric='hits@50')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        for N in [100]:
            neg_test_pred_N = neg_test_pred.view(-1, 100)
            for K in [1, 5, 10]:
                hits_evaluator.K = K
                test_hits = hits_evaluator.eval({
                    'y_pred_pos': pos_test_pred,
                    'y_pred_neg': neg_test_pred_N,
                })[f'hits@{K}']

                results[f'Hits@{K}@{N}'] = test_hits

    if 'mrr' in metric:
        mrr_evaluator = Evaluator(eval_metric='mrr')
        flag = normal_label
        pos_test_pred = torch.from_numpy(pos_out[flag == 1])
        neg_test_pred = torch.from_numpy(pos_out[flag == 0])

        neg_test_pred = neg_test_pred.view(-1, 100)

        mrr = mrr_evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })

        if isinstance(mrr, torch.Tensor):
            mrr = mrr.item()
        results['mrr'] = mrr
    return results


def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    r"""
    Computes the accuracy of prediction given the labels, and returns both f1 score for evaluation and testing.

    Args:
        emb(Tensor): The embedding of dataset.
        train_nids (Tensor): Index of the training set node.
        val_nids (Tensor): Index of the validating set node.
        test_nids (Tensor): Index of the testing set node.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test

