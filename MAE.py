"""
Modified from https://github.com/BeautyOfWeb/Multiview-AutoEncoder
"""
import torch
from torch import nn
from torch.nn import functional as F

class MAE(nn.Module):
    """
    Class for creating a Multi-view Factorization AutoEncoder(MAE)
    """

    def __init__(self, in_dims, d_dims, hidden_dims, in1, in2, fuse_type='sum'): #List of input dimensions, list of embedding dimensions, latent dimension, embedding network 1, embedding network 2, information fusion type('sum' or 'cat')
        super().__init__()
        self.num_views = len(in_dims)
        self.in_dims = in_dims
        self.d_dims = d_dims
        self.fuse_type = fuse_type
        self.hidden_dims = hidden_dims
        
        self.in1 = in1
        self.in2 = in2

        layers_one = [nn.Linear(d_dims[0], hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers_one.append(nn.ReLU())
            layers_one.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            #layers.Dropout(p=0.1)
        
        self.encoder_one = nn.Sequential(*layers_one)
        self.decoder_one = nn.Linear(hidden_dims[-1], d_dims[0])
        
        layers_two = [nn.Linear(d_dims[1], hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            layers_two.append(nn.ReLU())
            layers_two.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            #layers.Dropout(p=0.1)
        
        self.encoder_two = nn.Sequential(*layers_two)
        self.decoder_two = nn.Linear(hidden_dims[-1], d_dims[1])
        
        if self.fuse_type == 'sum':
            fuse_dim = hidden_dims[-1]
        elif self.fuse_type == 'cat':
            fuse_dim = hidden_dims[-1] * len(in_dims)
        else:
            raise ValueError(f"fuse_type should be 'sum' or 'cat', but is {fuse_type}")
        self.latent_size = fuse_dim
        
        #For survival: input dimension is self.latent_size
        
    def forward(self, xs):
        if isinstance(xs, torch.Tensor):
            xs = xs.split(self.in_dims, dim=1)
        assert len(xs) == self.num_views
        input_one = xs[0]
        input_two = xs[1]
        input_one = self.in1(input_one)
        input_two = self.in2(input_two)

        encoded_one = self.encoder_one(input_one)
        encoded_two = self.encoder_two(input_two)
        
        decoded_one = self.decoder_one(encoded_one)
        decoded_two = self.decoder_two(encoded_two)
        if self.fuse_type == 'sum':
            out = torch.stack([encoded_one, encoded_two], dim=-1).mean(dim=-1)
        else:
            out = torch.cat([encoded_one, encoded_two], dim=-1)
        latent_representation = out
        
        return torch.cat([decoded_one, decoded_two], dim=-1), torch.cat([encoded_one, encoded_two], dim=-1), torch.cat([input_one, input_two], dim=-1) #,latent_representation
        
    
def cosine_similarity(x, y=None, eps=1e-8):
  """Calculate cosine similarity between two matrices; 
  Args:
    x: N*p tensor
    y: M*p tensor or None; if None, set y = x
    This function do not broadcast
  
  Returns:
    N*M tensor
  """
  w1 = torch.norm(x, p=2, dim=1, keepdim=True)
  if y is None:
    w2 = w1.squeeze(dim=1)
    y = x
  else:
    w2 = torch.norm(y, p=2, dim=1)
  w12 = torch.mm(x, y.t())
  return w12 / (w1*w2).clamp(min=eps)


class Loss_view_similarity(nn.Module):
  r"""The input is a multi-view representation of the same set of patients, 
      i.e., a set of matrices with shape (num_samples, feature_dim). feature_dim can be different for each view
    This loss will penalize the inconsistency among different views.
    This is somewhat limited, because different views should have both shared and complementary information
      This loss only encourages the shared information across views, 
      which may or may not be good for certain applications.
    A trivial solution for this is multi-view representation are all the same; then loss -> -1
    The two loss_types 'circle' and 'hub' can be quite different and unstable.
      'circle' tries to make all feature representations across views have high cosine similarity,
      while 'hub' only tries to make feature representations within each view have high cosine similarity;
      by multiplying 'mean-feature' target with 'hub' loss_type, it might 'magically' capture both within-view and 
        cross-view similarity; set as default choice; but my limited experimental results do not validate this;
        instead, 'circle' and 'hub' are dominant, while explicit_target and cal_target do not make a big difference 
    Cosine similarity are used here; To do: other similarity metrics
  Args:
    sections: a list of integers (or an int); this is used to split the input matrix into chunks;
      each chunk corresponds to one view representation.
      If input xs is not a torch.Tensor, this will not be used; assume xs to be a list of torch.Tensors
      sections being an int implies all feature dim are the same, set sections = feature_dim, NOT num_sections!
    loss_type: supose there are three views x1, x2, x3; let s_ij = cos(x_i,x_j), s_i = cos(x_i,x_i)
      if loss_type=='cicle', similarity = s12*s23*target if fusion_type=='multiply'; s12+s23 if fusion_type=='sum'                   
        This is fastest but requires x1, x2, x3 have the same shape
      if loss_type=='hub', similarity=s1*s2*s3*target if fusion_type=='multiply'; 
        similarity=|s1|+|s2|+|s3|+|target| if fusion_type=='sum'
        Implicitly, target=1 (fusion_type=='multiply) or 0 (fusion_type=='sum') if explicit_target is False
        if graph_laplacian is False:
          loss = - similarity.abs().mean()
        else:
          s = similarity.abs(); L_s = torch.diag(sum(s, axis=1)) - s #graph laplacian
          loss = sum_i(x_i * L_s * x_i^T)
    explicit_target: if False, target=1 (fusion_type=='multiply) or 0 (fusion_type=='sum') implicitly
      if True, use given target or calculate it from xs
      # to do handle the case when we only use the explicitly given target
    cal_target: if 'mean-similarity', target = (cos(x1,x1) + cos(x2,x2) + cos(x3,x3))/3
                if 'mean-feature', x = (x1+x2+x3)/3; target = cos(x,x); this requires x1,x2,x3 have the same shape
    target: default None; only used when explicit_target is True
      This saves computation if target is provided in advance or passed as input
    fusion_type: if 'multiply', similarity=product(similarities); if 'sum', similarity=sum(|similarities|);
      work with loss_type
    graph_laplacian:  if graph_laplacian is False:
          loss = - similarity.abs().mean()
        else:
          s = similarity.abs(); L_s = torch.diag(sum(s, axis=1)) - s #graph laplacian
          loss = sum_i(x_i * L_s * x_i^T)
  Inputs:
    xs: a set of torch.Tensor matrices of (num_samples, feature_dim), 
      or a single matrix with self.sections being specified
    target: the target cosine similarity matrix; default None; 
      if not given, first check if self.targets is given; 
        if self.targets is None, then calulate it according to cal_target;
      only used when self.explicit_target is True
  Output:
    loss = -similarity.abs().mean() if graph_laplacian is False # Is this the right way to do it?
      = sum_i(x_i * L_s * x_i^T) if graph_laplacian is True # call get_interaction_loss()
    
  """
  def __init__(self, sections=None, loss_type='hub', explicit_target=False, 
    cal_target='mean-feature', target=None, fusion_type='multiply', graph_laplacian=False):
    super(Loss_view_similarity, self).__init__()
    self.sections = sections
    if self.sections is not None:
      if not isinstance(self.sections, int):
        assert len(self.sections) >= 2  
    self.loss_type = loss_type
    assert self.loss_type in ['circle', 'hub']
    self.explicit_target = explicit_target
    self.cal_target = cal_target
    self.target = target
    self.fusion_type = fusion_type
    self.graph_laplacian = graph_laplacian
    # I got nan losses easily for whenever graph_laplacian is True, especially the following case; did not know why
    # probably I need normalize similarity during every forward?
    assert not (fusion_type=='multiply' and graph_laplacian) and not (loss_type=='circle' and graph_laplacian)

  def forward(self, xs, target=None):
    if isinstance(xs, torch.Tensor):
      # make sure xs is a list of tensors corresponding to multiple views
      # this requires self.sections to valid
      xs = xs.split(self.sections, dim=1) 
    # assert len(xs) >= 2 # comment this to save time for many forward passes
    similarity = 1
    if self.loss_type == 'circle':
      # assert xs[i-1].shape == xs[i].shape
      # this saves computation
      similarity_mats = [cosine_similarity(xs[i-1], xs[i]) for i in range(1, len(xs))]
      similarity_mats = [(m+m.t())/2 for m in similarity_mats] # make it symmetric
    elif self.loss_type == 'hub':
      similarity_mats = [cosine_similarity(x) for x in xs]
    if self.fusion_type=='multiply':
      for m in similarity_mats:
        similarity = similarity * m # element multiplication ensures the larget value to be 1
    elif self.fusion_type=='sum':
      similarity = sum(similarity_mats) / len(similarity_mats) # calculate mean to ensure the largest value to be 1

    if self.explicit_target:
      if target is None:
        if self.target is None:
          if self.cal_target == 'mean-similarity':
            target = torch.stack(similarity_mats, dim=0).mean(0)
          elif self.cal_target == 'mean-feature':
            x = torch.stack(xs, -1).mean(-1) # the list of view matrices must have the same dimension
            target = cosine_similarity(x)
          else:
            raise ValueError(f'cal_target should be mean-similarity or mean-feature, but is {self.cal_target}')
        else:
          target = self.target
      if self.fusion_type=='multiply':
        similarity = similarity * target
      elif self.fusion_type=='sum':
        similarity = (len(similarity_mats)*similarity + target) / (len(similarity_mats) + 1) # Moving average
    similarity = similarity.abs() # ensure similarity to be non-negative
    if self.graph_laplacian:
      # Easily get nan loss when it is True; do not know why
      return sum([get_interaction_loss(similarity, w, loss_type='graph_laplacian', normalize=True) for w in xs]) / len(xs)
    else:
      return -similarity.mean() # to ensure the loss is within range [-1, 0]
    
