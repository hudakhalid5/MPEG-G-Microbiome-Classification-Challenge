# Docker Setup Guide for MPEG-G Microbiome Classification

This guide provides comprehensive instructions for setting up Docker on Windows for the centralized solution.

## 🐳 Docker Requirements

The centralized solution requires two specialized bioinformatics Docker containers:

1. **Genie** (`muefab/genie:latest`) - MPEG-G decompression (MGB → FASTQ)
2. **Jellyfish** (`quay.io/biocontainers/kmer-jellyfish:2.3.1--py310h184ae93_5`) - K-mer counting

## 📦 Installation Instructions

### Windows 10/11 Setup

#### 1. Install Docker Desktop
```bash
# Download Docker Desktop for Windows
# Visit: https://www.docker.com/products/docker-desktop

# Or use winget (Windows Package Manager)
winget install Docker.DockerDesktop
```

#### 2. Enable WSL2 Backend
- Docker Desktop → Settings → General
- Check "Use the WSL 2 based engine"
- Apply & Restart

#### 3. Configure Resources
- Docker Desktop → Settings → Resources
- **Memory**: Set to at least 4GB (8GB recommended)
- **CPU**: Use all available cores
- **Disk**: Allocate at least 10GB for images

## 🚀 Image Setup

### Pull Required Images

```bash
# Pull Genie image for MPEG-G decompression
docker pull muefab/genie:latest

# Pull Jellyfish image for k-mer counting
docker pull quay.io/biocontainers/kmer-jellyfish:2.3.1--py310h184ae93_5

# Verify images are downloaded
docker images | grep -E "(genie|jellyfish)"
```

### Verify Installation

#### Test Genie
```bash
# Test Genie help command
docker run --rm muefab/genie:latest help

# Expected output should include:
# ______           _
# / ____/__  ____  (_)__
# / / __/ _ \/ __ \/ / _ \
# / /_/ /  __/ / / / /  __/
# \____/\___/_/ /_/_/\___/
```

#### Test Jellyfish
```bash
# Test Jellyfish version
docker run --rm quay.io/biocontainers/kmer-jellyfish:2.3.1--py310h184ae93_5 jellyfish --version

# Expected output: jellyfish 2.3.1
```

## 🔧 Configuration

### Volume Mounting

The pipeline uses volume mounting to share files between the host and containers:

```python
# Example volume mount configuration
work_root = Path.cwd()
docker_mount = f"{work_root}:/work"

# Docker command with volume mount
docker_cmd = [
    "docker", "run", "--rm",
    "-v", docker_mount,  # Mount current directory
    "muefab/genie:latest", "run",
    "-i", "/work/input.mgb",
    "-o", "/work/output.fastq"
]
```

### Path Conversion

The pipeline includes automatic path conversion for Windows compatibility:

```python
def to_docker_path(local_path):
    """Convert Windows path to Docker container path"""
    abs_path = Path(local_path).resolve()
    rel_path = abs_path.relative_to(work_root)
    return "/work/" + str(rel_path).replace('\\', '/')
```

## 🛠️ Pipeline-Specific Issues

### 1. MGB File Processing Failures

**Error**: "Genie failed: file not found"

**Diagnosis**:
```bash
# Check file permissions in Windows
dir TrainFiles\*.mgb

# Verify volume mounting
docker run --rm -v "%cd%:/work" muefab/genie:latest ls /work/TrainFiles/
```

**Solutions**:
- Ensure MGB files are extracted correctly
- Check file paths are absolute
- Verify Docker has access to the directory

### 2. Jellyfish Hash Size Issues

**Error**: "Jellyfish count failed: hash size too small"

**Solutions**:
```python
# Increase hash size for large files
JELLYFISH_HASH_SIZE = "100M"  # Increase from 50M

# Or use automatic sizing
JELLYFISH_HASH_SIZE = "500M"  # For very large datasets
```

### 3. Temporary File Cleanup

**Issue**: Temporary files not cleaned up after errors

**Solutions**:
```python
try:
    # Process files
    result = stream_mgb_to_kmers(mgb_file, temp_dir)
finally:
    # Ensure cleanup
    temp_fastq.unlink(missing_ok=True)
    temp_jf.unlink(missing_ok=True)
```

## ✅ Validation Checklist

Before running the pipeline, verify:

- [ ] Docker Desktop is running and functioning
- [ ] Both required images are pulled successfully
- [ ] Test commands return expected output
- [ ] Volume mounting works with your project directory
- [ ] Sufficient memory allocated (8GB+ recommended)
- [ ] MGB files are accessible in the project directory
- [ ] No firewall blocking Docker registry access

## 📋 Quick Reference

### Essential Commands

```bash
# Pull images
docker pull muefab/genie:latest
docker pull quay.io/biocontainers/kmer-jellyfish:2.3.1--py310h184ae93_5

# Test installation
docker run --rm muefab/genie:latest help
docker run --rm quay.io/biocontainers/kmer-jellyfish:2.3.1--py310h184ae93_5 jellyfish --version

# Clean up
docker system prune -a
```