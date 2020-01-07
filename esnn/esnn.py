from collections import defaultdict
import numpy as np

class Neuron():
    def __init__(self, w, t, label, merge_count=1, psp=0):
        self.w = w
        self.t = t
        self.label = label
        self.merge_count = merge_count
        self.psp = psp

    def update(self, w, t):
        self.w = (self.merge_count * self.w + w) / (1 + self.merge_count)
        self.t = (self.merge_count * self.t + t) / (1 + self.merge_count)
        self.merge_count += 1


class ESNN():
    def __init__(self, encoder, m=0.9, c=0.7, s=0.6):
        self.encoder = encoder
        self.repository = defaultdict(list)
        self.m = m
        self.c = c
        self.s = s

    def train(self, samples, labels):
        for i, sample in enumerate(samples):
            label = labels[i]
            spikes = self.encoder(sample)
            index = np.argsort(spikes)
            w = np.zeros(len(spikes))
            w[index] = self.m ** (np.arange(len(spikes)))
            u_max = np.sum(w**2)
            theta = self.c * u_max

            similar_neuron = self._get_similar(w, label)

            if similar_neuron is None:
                self.repository[label].append(Neuron(w, theta, label))
            else:
                similar_neuron.update(w, theta)

    def test(self, samples):
        res = np.zeros(len(samples))

        self.all_neurons = []

        for neurons in self.repository.values():
            self.all_neurons.extend(neurons)

        self.w_matrix = np.zeros([len(self.all_neurons), self.all_neurons[0].w.shape[0]])
        self.all_theta = np.zeros(len(self.all_neurons))
        for i, neuron in enumerate(self.all_neurons):
            self.w_matrix[i,:] = neuron.w
            self.all_theta[i] = neuron.t

        pred = np.zeros(len(samples), dtype="int")
        for i, sample in enumerate(samples):
            spikes = self.encoder(sample)
            res = self.propagate(spikes)

            if res is None:
                pred[i] = -1
            else:
                pred[i] = self.all_neurons[res["idx"]].label

        return pred

    def propagate(self, spikes):
        self.all_psp = np.zeros(len(self.all_neurons))
        s = np.argsort(spikes)  # sorted by spike time
        for i, idx in np.ndenumerate(s):
            self.all_psp = self.all_psp + self.w_matrix[:, idx] * (self.m** i[0])
            active_neurons = np.argwhere(self.all_psp - self.all_theta > 0)

            if (len(active_neurons) > 0):
                neuron_idx = active_neurons[0][0]
                return {
                    "idx": neuron_idx,
                    "spike_time": spikes[idx]
                }
        return None

    def _get_similar(self, w, label):
        neurons = self.repository.get(label)
        if not neurons:
            return None

        dist = np.zeros(len(neurons))

        for i, neuron in enumerate(neurons):
            dist[i] = np.linalg.norm(w - neuron.w)

        min_dist_idx = np.argmin(dist)
        min_dist = dist[min_dist_idx]

        if min_dist < self.s:
            return neurons[min_dist_idx]
        else:
            return None
