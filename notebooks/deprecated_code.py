params = {x[0]: x[1] for x in t5_base_model.named_parameters()}
params['crf.modules_to_save.default._constraint_mask']


# print(*params.keys(), sep='\n')

aaa = torch.nn.Parameter(torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 1., 1., 0., 0., 0., 0., 1.],
        [0., 0., 1., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 1., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 1., 0., 0.],
        [1., 1., 0., 1., 1., 1., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0.]]), requires_grad=False)

t5_base_model.crf.original_module._constraint_mask, t5_base_model.crf.original_module.tensor_constraint_mask

t5_base_model.crf.modules_to_save.default._constraint_mask, t5_base_model.crf.modules_to_save.default.tensor_constraint_mask