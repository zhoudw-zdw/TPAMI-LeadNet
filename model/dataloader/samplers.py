import torch
import numpy as np
import itertools
from tqdm import tqdm    

class CategoriesSamplerWithMultiAttr():
    '''sample one or more concepts from multiple attributes'''
    def __init__(self, label, n_batch, n_cls, n_per, n_concept_range=3, dict_filter = None):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.label = label
        self.n_attr = self.label.shape[1]
        
        assert(n_concept_range <= self.n_attr)
        joint_label_set = []
        for i in range(1, n_concept_range+1):
            joint_label_set.extend(list(itertools.combinations(range(self.n_attr), i)))
        
        # check label validness
        joint_label_dict = {}
        for j_label in tqdm(joint_label_set):
            class_m_id = []
            current_label_set = self.label[:, np.array(j_label)]
            current_unique_label_set, current_label_index = np.unique(current_label_set, return_inverse=True, axis=0)
            num_sub_label = current_unique_label_set.shape[0]
            ind_set = [np.argwhere(current_label_index == e).reshape(-1) for e in range(num_sub_label)]
            assert(sum([e.shape[0] for e in ind_set]) == self.label.shape[0])
            for ind in ind_set:
                if ind.shape[0] > self.n_per * 2:
                    class_m_id.append(torch.from_numpy(ind))
            if len(class_m_id) > self.n_cls + 1:
                joint_label_dict[j_label]= class_m_id

        if dict_filter is not None:
            for e in set(joint_label_dict.keys()).difference(dict_filter):
                joint_label_dict.pop(e, None)

        self.joint_label_dict = joint_label_dict
        self.joint_label_set = sorted(list(joint_label_dict.keys()))
        self.num_concept = len(joint_label_dict)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # choose a concept
            concet_ind = torch.randperm(self.num_concept)[0]
            selected_concepts = self.joint_label_set[concet_ind]
            # choose classes
            classes = torch.randperm(len(self.joint_label_dict[selected_concepts]))[:self.n_cls]
            for c in classes:
                l = self.joint_label_dict[selected_concepts][c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            batch = torch.cat([batch.view(-1, 1), concet_ind * torch.ones_like(batch).view(-1, 1)], 1)
            yield batch


class CategoriesSamplerWithAttr():
    '''sample from multiple attributes'''
    def __init__(self, label, n_batch, n_cls, n_per, attr_filter=None):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        
        if attr_filter is not None:
            label = label[:, attr_filter]
        self.label = label
        self.n_attr = self.label.shape[1]
        
        self.m_ind = []
        for i in range(self.n_attr):
            class_m_id = []
            for j in np.unique(self.label[:, i]):
                ind = np.argwhere(label[:, i] == j).reshape(-1)
                ind = torch.from_numpy(ind)
                class_m_id.append(ind)
            self.m_ind.append(class_m_id)

        # check data
        for i in range(self.n_attr):
            for j in range(len(self.m_ind[i])):
                assert(len(self.m_ind[i][j]) >= self.n_per)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # choose an attribute
            attr_ind = torch.randperm(self.n_attr)[0]
            # choose classes
            classes = torch.randperm(len(self.m_ind[attr_ind]))[:self.n_cls]
            for c in classes:
                l = self.m_ind[attr_ind][c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            batch = torch.cat([batch.view(-1, 1), attr_ind * torch.ones_like(batch).view(-1, 1)], 1)
            yield batch


class CategoriesSamplerSameClass():

    def __init__(self, label, n_batch, n_cls, n_per):
        '''Sample all images from the same set of classes'''
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        classes = torch.randperm(len(self.m_ind))[:self.n_cls]        
        for i_batch in range(self.n_batch):
            batch = []
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class CategoriesSamplerMix():
    # able to sample sub-tasks
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # random select the number of classes
            classes = torch.randperm(len(self.m_ind))[:(np.random.choice(range(1, self.n_cls))+1)]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch



class CategoriesSamplerMix2():
    # able to sample sub-tasks in a range
    def __init__(self, label, n_batch, n_cls1, n_cls2, n_per):
        self.n_batch = n_batch
        self.n_cls1 = n_cls1
        self.n_cls2 = n_cls2
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # random select the number of classes
            classes = torch.randperm(len(self.m_ind))[:(np.random.choice(range(self.n_cls1, self.n_cls2))+1)]
            if len(classes) < self.n_cls2:
                # pad classes with initial points
                classes = torch.cat([classes, classes[:self.n_cls2 - len(classes)]], 0)
                                    
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            
            
            
class CategoriesSamplerMix3():
    # sample 3 tasks at a time, the first two come from the same classes
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        # random select the number of classes
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls * 2]
            classes = torch.cat([classes[:self.n_cls], classes[:self.n_cls], classes[self.n_cls:]])
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch