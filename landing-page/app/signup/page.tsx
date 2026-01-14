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
import { Zap, Mail, Lock, User, Eye, EyeOff, Check } from 'lucide-react'
import { Button } from '@/components/ui/Button'

export default function SignupPage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [agreedToTerms, setAgreedToTerms] = useState(false)

  const passwordStrength = [
    { label: '8+ chars', met: password.length >= 8 },
    { label: 'Number', met: /\d/.test(password) },
    { label: 'Upper', met: /[A-Z]/.test(password) },
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!agreedToTerms) {
      alert('Please agree to the terms and conditions')
      return
    }
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
      alert('Signup functionality would be implemented with your auth provider')
    }, 1500)
  }

  return (
    <main className="min-h-screen flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-xs px-4"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2">
            <div className="w-10 h-10 bg-gradient-to-br from-primary via-purple-500 to-accent rounded-xl flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
          </Link>
          <span className="text-xl font-bold gradient-text ml-2">KERNELIZE</span>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-xs font-semibold text-white mb-1.5">Full Name</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className="h-3.5 w-3.5 text-gray-400" />
              </div>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full pl-9 pr-3 py-2.5 bg-white rounded-full text-gray-900 placeholder-gray-400 text-sm focus:outline-none"
                placeholder="John Doe"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-white mb-1.5">Email</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Mail className="h-3.5 w-3.5 text-gray-400" />
              </div>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-9 pr-3 py-2.5 bg-white rounded-full text-gray-900 placeholder-gray-400 text-sm focus:outline-none"
                placeholder="name@company.com"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-white mb-1.5">Password</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-3.5 w-3.5 text-gray-400" />
              </div>
              <input
                type={showPassword ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-9 pr-9 py-2.5 bg-white rounded-full text-gray-900 placeholder-gray-400 text-sm focus:outline-none"
                placeholder="Create password"
                required
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                {showPassword ? <EyeOff className="h-3.5 w-3.5 text-gray-400" /> : <Eye className="h-3.5 w-3.5 text-gray-400" />}
              </button>
            </div>
          </div>

          {password.length > 0 && (
            <div className="flex gap-2 text-xs">
              {passwordStrength.map((req, i) => (
                <span key={i} className="flex items-center gap-0.5">
                  <Check className={`h-2.5 w-2.5 ${req.met ? 'text-green-400' : 'text-gray-500'}`} />
                  <span className={req.met ? 'text-green-400' : 'text-gray-500'}>{req.label}</span>
                </span>
              ))}
            </div>
          )}

          <label className="flex items-start gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={agreedToTerms}
              onChange={(e) => setAgreedToTerms(e.target.checked)}
              className="mt-0.5 w-3.5 h-3.5 rounded border-gray-300"
            />
            <span className="text-xs text-gray-400">
              I agree to{' '}
              <Link href="/terms" className="text-primary hover:text-accent">Terms</Link>
              {' '}and{' '}
              <Link href="/privacy" className="text-primary hover:text-accent">Privacy</Link>
            </span>
          </label>

          <Button
            type="submit"
            className="w-full py-2 rounded-full text-sm"
            disabled={isLoading || !agreedToTerms}
          >
            {isLoading ? 'Creating...' : 'Create Account'}
          </Button>
        </form>

        <p className="text-center text-xs text-gray-400 mt-5">
          Already have an account?{' '}
          <Link href="/login" className="text-primary hover:text-accent font-medium">
            Sign in
          </Link>
        </p>
      </motion.div>
    </main>
  )
}
