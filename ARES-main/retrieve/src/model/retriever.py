import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing


class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())

        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index
    ):
        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)

        return result_list


class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs
    ):
        super().__init__()

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)

        pred_in_size = 4 * emb_size
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (
            DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def encode_entity_features(
        self,
        graph_h_id_tensor,
        graph_t_id_tensor,
        entity_embs,
        num_non_text_entities,
        topic_entity_one_hot
    ):
        device = entity_embs.device

        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ],
            dim=0,
        )
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            graph_h_id_tensor,
            graph_t_id_tensor,
        ], dim=0)
        reverse_edge_index = torch.stack([
            graph_t_id_tensor,
            graph_h_id_tensor,
        ], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)

        return torch.cat(h_e_list, dim=1)

    def compose_triple_inputs(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        relation_embs,
        entity_features
    ):
        h_r = relation_embs[r_id_tensor]
        return torch.cat([
            q_emb.expand(len(h_r), -1),
            entity_features[h_id_tensor],
            h_r,
            entity_features[t_id_tensor],
        ], dim=1)

    def encode_triple_inputs(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        graph_h_id_tensor=None,
        graph_t_id_tensor=None
    ):
        if graph_h_id_tensor is None:
            graph_h_id_tensor = h_id_tensor
        if graph_t_id_tensor is None:
            graph_t_id_tensor = t_id_tensor

        entity_features = self.encode_entity_features(
            graph_h_id_tensor=graph_h_id_tensor,
            graph_t_id_tensor=graph_t_id_tensor,
            entity_embs=entity_embs,
            num_non_text_entities=num_non_text_entities,
            topic_entity_one_hot=topic_entity_one_hot,
        )

        return self.compose_triple_inputs(
            h_id_tensor=h_id_tensor,
            r_id_tensor=r_id_tensor,
            t_id_tensor=t_id_tensor,
            q_emb=q_emb,
            relation_embs=relation_embs,
            entity_features=entity_features,
        )

    def predict_logits(self, x_stage1):
        return self.pred(x_stage1)

    def forward_with_features(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        graph_h_id_tensor=None,
        graph_t_id_tensor=None
    ):
        x_stage1 = self.encode_triple_inputs(
            h_id_tensor=h_id_tensor,
            r_id_tensor=r_id_tensor,
            t_id_tensor=t_id_tensor,
            q_emb=q_emb,
            entity_embs=entity_embs,
            num_non_text_entities=num_non_text_entities,
            relation_embs=relation_embs,
            topic_entity_one_hot=topic_entity_one_hot,
            graph_h_id_tensor=graph_h_id_tensor,
            graph_t_id_tensor=graph_t_id_tensor,
        )
        stage1_logit = self.predict_logits(x_stage1)
        stage1_prob = torch.sigmoid(stage1_logit)

        return {
            'x_stage1': x_stage1,
            'stage1_logit': stage1_logit,
            'stage1_prob': stage1_prob,
        }

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        x_stage1 = self.encode_triple_inputs(
            h_id_tensor=h_id_tensor,
            r_id_tensor=r_id_tensor,
            t_id_tensor=t_id_tensor,
            q_emb=q_emb,
            entity_embs=entity_embs,
            num_non_text_entities=num_non_text_entities,
            relation_embs=relation_embs,
            topic_entity_one_hot=topic_entity_one_hot,
        )

        return self.predict_logits(x_stage1)
