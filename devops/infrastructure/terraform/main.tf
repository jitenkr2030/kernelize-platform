# KERNELIZE Platform - Infrastructure as Code
# Multi-cloud deployment with Terraform

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = "kernelize-terraform-state"
    key    = "infrastructure/terraform.tfstate"
    region = "us-east-1"
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "kernelize-cluster"
}

variable "node_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 3
}

variable "instance_types" {
  description = "EC2 instance types for worker nodes"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

# Provider configuration
provider "aws" {
  region = var.region
}

# VPC Configuration
module "vpc" {
  source = "./modules/vpc"
  
  name = "kernelize-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.region}a", "${var.region}b", "${var.region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = true
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # EKS managed node group
  node_groups = {
    main = {
      desired_capacity = var.node_count
      max_capacity     = 10
      min_capacity     = 1
      
      instance_types = var.instance_types
      
      capacity_type = "SPOT"
      
      k8s_labels = {
        Environment = var.environment
        Project     = "KERNELIZE"
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

# RDS Database
resource "aws_db_instance" "kernelize_db" {
  identifier = "kernelize-${var.environment}"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "kernelize"
  username = "kernelize"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.kernelize.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  multi_az               = true
  publicly_accessible    = false
  skip_final_snapshot    = false
  deletion_protection    = true
  
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "kernelize_redis" {
  replication_group_id         = "kernelize-${var.environment}"
  description                  = "KERNELIZE Redis cluster"
  
  port                        = 6379
  parameter_group_name        = "default.redis7"
  node_type                   = "cache.t3.micro"
  num_cache_clusters          = 2
  automatic_failover_enabled  = true
  multi_az_enabled           = true
  
  subnet_group_name = aws_elasticache_subnet_group.kernelize.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth_token.result
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "kernelize_storage" {
  bucket = "kernelize-${var.environment}-storage"
}

resource "aws_s3_bucket" "kernelize_logs" {
  bucket = "kernelize-${var.environment}-logs"
}

resource "aws_s3_bucket" "kernelize_backups" {
  bucket = "kernelize-${var.environment}-backups"
}

# S3 Bucket Encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "storage" {
  bucket = aws_s3_bucket.kernelize_storage.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Security Groups
resource "aws_security_group" "eks" {
  name_prefix = "kernelize-eks-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "kernelize-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks.id]
  }
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "kernelize-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks.id]
  }
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

# Subnet Groups
resource "aws_db_subnet_group" "kernelize" {
  name       = "kernelize-${var.environment}"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Environment = var.environment
    Project     = "KERNELIZE"
  }
}

resource "aws_elasticache_subnet_group" "kernelize" {
  name       = "kernelize-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

# Random passwords
resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "random_password" "redis_auth_token" {
  length  = 32
  special = true
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}