import torch 
import torch.nn as nn 

class FeedForwardResidualBlock(nn.Module):
    def __init__(self, dim, expansion_multiplier=1):
        super(FeedForwardResidualBlock, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim, dim * expansion_multiplier),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim * expansion_multiplier),
            nn.Linear(dim * expansion_multiplier, dim)
        )

    def forward(self, x):
        return x + self.projection(x)
    
class panel_SCL(nn.Module): 
    """
    Inputs: x, (bs, 3, 160, 160), e.g., (1, 3, 160, 160)
    Outputs: CNN feature, (bs, 3, 80)
    """
    def __init__(self): 
        super().__init__()
        
        self.conv = nn.Sequential( # no MaxPool
            nn.Conv2d(1, 16, 3, padding=1, stride=2), nn.BatchNorm2d(16), nn.ReLU(inplace=True), # [3, 16, 80, 80]
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), # [3, 16, 40, 40]
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), # [3, 32, 20, 20]
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)) # [3, 32, 10, 10]

        self.conv_projection = nn.Sequential(nn.Linear(6400, 80),nn.ReLU(inplace=True), FeedForwardResidualBlock(80))
        
        self.attribute_network = nn.Sequential(nn.Linear(256, 128),nn.ReLU(inplace=True),nn.Linear(128, 8))
        self.attribute_residual = FeedForwardResidualBlock(80)
        
    def forward(self, x): 
        batch_size, num_panels, h, w = x.size() # [1, 3, 160, 160]
        bs = batch_size * num_panels
        x = x.view(bs, 1, h, w) # [3, 1, 160, 160]
        
        x = self.conv(x) # [3, 32, 80, 80]  
        x = x.view(bs, 32, -1) # [3, 32, 6400]
        x = self.conv_projection(x) # [3, 32, 80]
        
        x = x.reshape(bs, 32, 10, 8).transpose(2,1).reshape(bs, 10, -1) # scatter, [3, 10, 256]
        x = self.attribute_network(x).view(bs, 80) # [3, 10, 8] --> [3, 80]
        x = self.attribute_residual(x).view(batch_size, num_panels, 80) # torch.Size([3, 80])
        return x 
    
class relation_SCL(nn.Module): 
    """
    Inputs: x, (bs, num_panels, dim_panels), e.g., torch.Size([13, 3, 80])
    Outputs: feature_relation, (bs, dim_relation), e.g., torch.Size([13, 400])
    """
    def __init__(self): 
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=True))
        self.conv3 = nn.Linear(32, 5)
        
        self.fc = FeedForwardResidualBlock(400)
        
    def forward(self, x): 
        batch_size, num_panels, dim_panel = x.size()
        x = x.transpose(-2, -1) # [1, 80, 3]
        
        x1 = self.conv1(x) # [1, 80, 64]
        x2 = self.conv2(x1) # [1, 80, 32]
        x3 = self.conv3(x2) # [1, 80, 5]
        x3 = x3.flatten(start_dim=1) # [1, 400]
        
        x4 = self.fc(x3)
        return x4
    
class relation_RN(nn.Module): 
    """
    Inputs: x, (bs, num_panels, dim_panels), e.g., torch.Size([13, 3, 80])
    Outputs: feature_relation, (bs, dim_relation), e.g., torch.Size([13, 400])
    """
    def __init__(self): 
        super().__init__()
        
        self.proj = nn.Sequential(nn.Linear(83, 256), nn.ReLU())
        
        self.g = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), 
                nn.Linear(512, 512), nn.ReLU(), 
                nn.Linear(512, 512), nn.ReLU(), 
                nn.Linear(512, 256),  nn.ReLU())
        
        self.fc1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
        self.fc3 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 400)) 
        
        self.tags = torch.eye(3).unsqueeze(0)
        self.tags.requires_grad = False 
        
    def forward(self, x): 
        tags = self.tags.to(x.device).repeat(x.shape[0], 1, 1)
        x1 = torch.cat((x, tags), dim=2) # [13, 3, 83]
        x2 = self.proj(x1) # [13, 3, 256]
        x3 = torch.cat((x2.unsqueeze(1).expand(-1, 3, -1, -1),
                       x2.unsqueeze(2).expand(-1, -1, 3, -1)),dim=3).view(-1, 9, 512)# [13, 9, 512]
        x4 = self.g(x3).sum(1) # [13, 256]
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)
        x7 = self.fc3(x6)
        return x7
    
class relation_conv(nn.Module): 
    """
    1) kernel2: channel_list=[64, 32, 5], kernel=2, dim=385
    2) kernel4: channel_list=[64, 32, 5], kernel=4, dim=355
    3) chan32: channel_list=[32, 32, 32], kernel=1, dim=2560
    4) chan16: channel_list=[16, 32, 64], kernel=1, dim=5120
    """
    def __init__(self, channel_list=[64, 32, 5], kernel=2, dim=385): 
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=channel_list[0], kernel_size=kernel)
        self.conv2 = nn.Conv1d(in_channels=channel_list[0], out_channels=channel_list[1], kernel_size=kernel)
        self.conv3 = nn.Conv1d(in_channels=channel_list[1], out_channels=channel_list[2], kernel_size=kernel)
        
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Sequential(nn.Linear(dim, 400), nn.ReLU(inplace=True))
        
        self.fc = FeedForwardResidualBlock(400)
        
    def forward(self, x): 
        x1 = self.relu(self.conv1(x)) # [1, 64, 79]
        x2 = self.relu(self.conv2(x1)) # [1, 32, 78]
        x3 = self.conv3(x2) # [1, 5, 77]
        x4 = x3.transpose(1,2).flatten(start_dim=1) # [1, 385]
        x5 = self.linear(x4)
        x6 = self.fc(x5) # torch.Size([13, 400])
        return x6
    
class SCL(nn.Module): 
    """
    Input: x: (batch_size, 3, 160, 160)
    Output: feature: (batch_size, 400)
    """
    def __init__(self, module_panel=panel_SCL(), module_relation=relation_SCL()):
        super(SCL, self).__init__()
        self.module_panel = module_panel 
        self.module_relation = module_relation 
        
    def forward(self, x):
        x = self.module_panel(x) #[batch_size, num_panel_total, dim_image], # torch.Size([13, 3, 80])
        feature = self.module_relation(x) # [batch_size, num_sets, dim_structure]
        return feature
    
class SCL_supervised(nn.Module): 
    """
    Input: x: (batch_size, 3, 160, 160)
    Output: outputs: (batch_size, num_classes); feature: (batch_size, 400)
    """
    def __init__(self, encoder=SCL(), dim_feature=400, m=35):
        super(SCL_supervised, self).__init__()
        
        self.encoder = encoder 
        self.readout = nn.Linear(dim_feature, m)
        
    def forward(self, x): 
        feature = self.encoder(x)
        outputs = self.readout(feature)
        return outputs, feature
    
### transformer 
from einops import rearrange, repeat
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class relation_Transformer(nn.Module):
    def __init__(self, dim=256, 
                 depth=6, 
                 heads=16, 
                 mlp_dim=256, 
                 dim_head=16, 
                 dropout=0.1, 
                 emb_dropout=0.1):
        super().__init__()
        
        self.proj = nn.Linear(80, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(nn.Linear(dim, 400), nn.ReLU(inplace=True), 
                                      nn.Linear(400, 400))

    def forward(self, x):
        x = self.proj(x)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)

        x = x.mean(dim = 1)
        x = self.mlp_head(x)
        return x