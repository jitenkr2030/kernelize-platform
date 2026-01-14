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
    { label: '8+ characters', met: password.length >= 8 },
    { label: 'Number', met: /\d/.test(password) },
    { label: 'Uppercase', met: /[A-Z]/.test(password) },
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

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-semibold text-white mb-2">Full Name</label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <User className="h-4 w-4 text-gray-400" />
              </div>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full pl-11 pr-4 py-3 bg-white rounded-full text-gray-900 placeholder-gray-400 text-sm focus:outline-none"
                placeholder="John Doe"
                required
              />
            </div>
          </div>

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
                placeholder="Create password"
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

          {password.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {passwordStrength.map((req, i) => (
                <span key={i} className="flex items-center gap-1 text-xs">
                  <Check className={`h-3 w-3 ${req.met ? 'text-green-400' : 'text-gray-500'}`} />
                  <span className={req.met ? 'text-green-400' : 'text-gray-400'}>{req.label}</span>
                </span>
              ))}
            </div>
          )}

          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={agreedToTerms}
              onChange={(e) => setAgreedToTerms(e.target.checked)}
              className="mt-0.5 w-4 h-4 rounded border-gray-300"
            />
            <span className="text-sm text-gray-400">
              I agree to the{' '}
              <Link href="/terms" className="text-primary hover:text-accent">Terms of Service</Link>
              {' '}and{' '}
              <Link href="/privacy" className="text-primary hover:text-accent">Privacy Policy</Link>
            </span>
          </label>

          <Button
            type="submit"
            className="w-full py-3 rounded-full"
            disabled={isLoading || !agreedToTerms}
          >
            {isLoading ? 'Creating account...' : 'Create Account'}
          </Button>
        </form>

        <p className="text-center text-sm text-gray-400 mt-6">
          Already have an account?{' '}
          <Link href="/login" className="text-primary hover:text-accent font-medium">
            Sign in
          </Link>
        </p>
      </motion.div>
    </main>
  )
}
