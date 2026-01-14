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

import { useState } from 'react'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { Zap, Mail, Lock, Eye, EyeOff } from 'lucide-react'
import { Button } from '@/components/ui/Button'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
      alert('Login functionality would be implemented with your auth provider')
    }, 1500)
  }

  return (
    <main className="min-h-screen flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-sm mx-4"
      >
        {/* Logo */}
        <div className="text-center mb-10">
          <Link href="/" className="inline-flex items-center gap-2">
            <div className="w-12 h-12 bg-gradient-to-br from-primary via-purple-500 to-accent rounded-xl flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
          </Link>
          <span className="text-2xl font-bold gradient-text ml-2">KERNELIZE</span>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-semibold text-white mb-2">Email</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Mail className="h-4 w-4 text-gray-400" />
              </div>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-11 pr-4 py-3 bg-white rounded-full text-gray-900 placeholder-gray-400 text-sm focus:outline-none"
                placeholder="name@company.com"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-white mb-2">Password</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Lock className="h-4 w-4 text-gray-400" />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-11 pr-11 py-3 bg-white rounded-full text-gray-900 placeholder-gray-400 text-sm focus:outline-none"
                placeholder="Enter password"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-4 flex items-center"
              >
                {showPassword ? <EyeOff className="h-4 w-4 text-gray-400" /> : <Eye className="h-4 w-4 text-gray-400" />}
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                className="w-4 h-4 rounded border-gray-300"
              />
              <span className="text-sm text-gray-300">Remember me</span>
            </label>
            <Link href="/forgot-password" className="text-sm text-primary hover:text-accent">
              Forgot password?
            </Link>
          </div>

          <Button
            type="submit"
            className="w-full py-3 rounded-full"
            disabled={isLoading}
          >
            {isLoading ? 'Signing in...' : 'Sign In'}
          </Button>
        </form>

        <p className="text-center text-sm text-gray-400 mt-6">
          Don't have an account?{' '}
          <Link href="/signup" className="text-primary hover:text-accent font-medium">
            Sign up
          </Link>
        </p>

        <p className="text-center text-xs text-gray-500 mt-6">
          By signing in, you agree to our{' '}
          <Link href="/terms" className="text-gray-400 hover:text-white">Terms</Link>
          {' '}and{' '}
          <Link href="/privacy" className="text-gray-400 hover:text-white">Privacy Policy</Link>
        </p>
      </motion.div>
    </main>
  )
}
