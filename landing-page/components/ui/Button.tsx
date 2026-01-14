/**
 * KERNELIZE Platform - Landing Page
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

'use client'

import { cn } from '@/lib/utils'
import { ButtonHTMLAttributes, forwardRef } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline'
  size?: 'sm' | 'md' | 'lg'
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', children, ...props }, ref) => {
    const baseStyles = 'inline-flex items-center justify-center font-semibold rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden'

    const variants = {
      primary: 'bg-gradient-to-r from-primary via-purple-500 to-accent text-white shadow-lg shadow-primary/25 hover:shadow-primary/40 hover:scale-105 focus:ring-primary',
      secondary: 'glass border border-white/10 text-text-primary hover:bg-white/10 hover:border-white/20 hover:shadow-lg hover:shadow-primary/10 focus:ring-white/20',
      ghost: 'text-text-secondary hover:text-text-primary hover:bg-white/5 focus:ring-white/20',
      outline: 'border-2 border-primary/50 text-primary hover:bg-primary/10 hover:border-primary focus:ring-primary',
    }

    const sizes = {
      sm: 'text-sm py-2.5 px-5',
      md: 'text-base py-3 px-6',
      lg: 'text-lg py-4 px-8',
    }

    return (
      <button
        ref={ref}
        className={cn(baseStyles, variants[variant], sizes[size], className)}
        {...props}
      >
        {/* Shimmer effect for primary button */}
        {variant === 'primary' && (
          <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full animate-shimmer" />
        )}
        {children}
      </button>
    )
  }
)

Button.displayName = 'Button'

export { Button }
export type { ButtonProps }
