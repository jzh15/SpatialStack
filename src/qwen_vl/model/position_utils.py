import torch


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal position embeddings.

    Args:
        embed_dim: Embedding dimension (must be even)
        pos: Position tensor of shape [H, W] or [N]

    Returns:
        emb: Position embeddings of shape [H*W, embed_dim] (or [N, embed_dim])
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # [embed_dim//2]

    pos = pos.flatten()
    out = torch.einsum("m,d->md", pos, omega)  # [N, embed_dim//2]

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # [N, embed_dim]
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embeddings from grid.

    Args:
        embed_dim: Embedding dimension (must be even)
        grid: Grid coordinates of shape [2, H, W]

    Returns:
        pos_embed: Position embeddings of shape [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [H*W, embed_dim//2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [H*W, embed_dim//2]
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_2d_sincos_pos_embed(height: int, width: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """
    Generate 2D sinusoidal position embeddings.

    Args:
        height: Height of the grid
        width: Width of the grid
        embed_dim: Embedding dimension (must be even)
        device: Device to create tensor on

    Returns:
        pos_embed: Position embeddings of shape [height*width, embed_dim]
    """
    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # [2, H, W]
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
