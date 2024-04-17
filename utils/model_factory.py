def get_model(model_name, args):
    name = model_name.lower()
    if name=="simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name=="finetune":
        from models.adam_finetune import Learner
        return Learner(args)
    elif name=="adapter":
        from models.SSITA_adapter import Learner
        return Learner(args)
    else:
        assert 0
