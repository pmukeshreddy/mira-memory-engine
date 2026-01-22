# =============================================================================
# Mira Memory Engine - Terraform Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN"
  value       = aws_ecs_cluster.main.arn
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Application Load Balancer zone ID"
  value       = aws_lb.main.zone_id
}

output "api_ecr_repository_url" {
  description = "ECR repository URL for API"
  value       = aws_ecr_repository.api.repository_url
}

output "frontend_ecr_repository_url" {
  description = "ECR repository URL for frontend"
  value       = aws_ecr_repository.frontend.repository_url
}

output "api_target_group_arn" {
  description = "API target group ARN"
  value       = aws_lb_target_group.api.arn
}

output "secrets_manager_arn" {
  description = "Secrets Manager secret ARN for API keys"
  value       = aws_secretsmanager_secret.api_keys.arn
}

output "cloudwatch_log_groups" {
  description = "CloudWatch log group names"
  value = {
    api      = aws_cloudwatch_log_group.api.name
    frontend = aws_cloudwatch_log_group.frontend.name
  }
}

# =============================================================================
# Connection Strings
# =============================================================================

output "api_url" {
  description = "URL to access the API"
  value       = "http://${aws_lb.main.dns_name}"
}

output "health_check_url" {
  description = "Health check endpoint"
  value       = "http://${aws_lb.main.dns_name}/api/v1/health"
}

# =============================================================================
# Deployment Commands
# =============================================================================

output "ecr_login_command" {
  description = "Command to login to ECR"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

output "push_api_image_commands" {
  description = "Commands to build and push API image"
  value       = <<-EOT
    docker build -t ${aws_ecr_repository.api.repository_url}:latest .
    docker push ${aws_ecr_repository.api.repository_url}:latest
    aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${var.project_name}-api --force-new-deployment
  EOT
}
