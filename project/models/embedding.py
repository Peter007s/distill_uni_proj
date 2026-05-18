def get_embedding_dim(name):
    dims = {
        'cifar10_resnet56': 64,
        'resnet20': 64,
        'resnet32': 64,
        'resnet50': 2048,
        'resnet18': 512,
        'mobilenet_v3_small': 576,
        'shufflenet_v2_x0_5': 1024,
        'squeezenet1_0': 512,
        'custom': 64
    }
    return dims.get(name, 64)

def extract_embedding(model, x, detach=True):
    if hasattr(model, 'get_embedding'):
        return model.get_embedding(x)

    emb = None
    def hook(module, input, output):
        nonlocal emb
        if detach:
            emb = output.detach()
        else:
            emb = output
        if emb.dim() == 4:
            emb = emb.squeeze(-1).squeeze(-1)
    
    layer = model.avgpool if hasattr(model, 'avgpool') else model.classifier
    h = layer.register_forward_hook(hook)
    model(x)
    h.remove()
    return emb