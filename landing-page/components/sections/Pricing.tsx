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
import { motion } from 'framer-motion'
import { Check, Sparkles, ArrowRight, Zap, Shield, Globe, Crown } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
import Link from 'next/link'

const plans = [
  {
    name: 'Starter',
    price: 0,
    description: 'Perfect for individual developers and small projects',
    gradient: 'from-slate-500 to-slate-600',
    features: [
      'Up to 1GB bandwidth/month',
      '1 custom domain',
      'Basic analytics',
      'Community support',
      '5 API endpoints',
      '7-day data retention',
    ],
    cta: 'Start Free',
    popular: false,
    limit: 'Free forever',
  },
  {
    name: 'Pro',
    price: 49,
    description: 'Ideal for growing teams and production applications',
    gradient: 'from-primary to-purple-500',
    features: [
      'Up to 100GB bandwidth/month',
      'Unlimited custom domains',
      'Advanced analytics & reports',
      'Priority email & chat support',
      'Unlimited API endpoints',
      '30-day data retention',
      'AI compression (10GB/month)',
      'Custom rate limits',
    ],
    cta: 'Start Free Trial',
    popular: true,
    limit: '30-day trial',
  },
  {
    name: 'Enterprise',
    price: 299,
    description: 'For large organizations with advanced requirements',
    gradient: 'from-amber-500 to-orange-500',
    features: [
      'Unlimited bandwidth',
      'Unlimited custom domains',
      'Custom analytics & dashboards',
      '24/7 phone & dedicated support',
      '99.99% SLA guarantee',
      'Unlimited data retention',
      'AI compression (unlimited)',
      'Custom rate limits',
      'On-premise deployment',
      'SSO & SAML integration',
      'Dedicated success manager',
    ],
    cta: 'Contact Sales',
    popular: false,
    limit: 'Custom pricing',
  },
]

const faqs = [
  {
    question: 'Can I change plans later?',
    answer: 'Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately, and we\'ll prorate any payments.',
  },
  {
    question: 'What payment methods do you accept?',
    answer: 'We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and wire transfers for Enterprise plans.',
  },
  {
    question: 'Is there a free trial?',
    answer: 'Yes! The Pro plan comes with a 30-day free trial. No credit card required to start. You can also use the Starter plan free forever.',
  },
  {
    question: 'What happens after my trial ends?',
    answer: 'After your trial, you can choose to upgrade to a paid plan or continue with the free Starter plan. Your data is never deleted.',
  },
]

export default function Pricing() {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'yearly'>('monthly')
  const [openFaq, setOpenFaq] = useState<number | null>(0)

  return (
    <section id="pricing" className="section-padding relative">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-primary/3 via-transparent to-transparent" />
      <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

      <div className="container-custom relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 glass rounded-full mb-6">
            <Sparkles className="w-4 h-4 text-accent" />
            <span className="text-sm text-text-secondary">Simple, Transparent Pricing</span>
          </div>
          <h2 className="section-title">
            Choose the perfect plan for
            <span className="block gradient-text mt-2">your scaling needs</span>
          </h2>
          <p className="section-subtitle mx-auto">
            Start free, scale as you grow. No hidden fees, no surprises.
          </p>

          {/* Billing toggle */}
          <div className="flex items-center justify-center gap-4 mt-8">
            <span className={`text-sm font-medium ${billingPeriod === 'monthly' ? 'text-text-primary' : 'text-text-muted'}`}>
              Monthly
            </span>
            <button
              onClick={() => setBillingPeriod(billingPeriod === 'monthly' ? 'yearly' : 'monthly')}
              className="relative w-14 h-7 rounded-full bg-white/10 border border-white/20 transition-colors hover:border-primary/50"
            >
              <motion.div
                className="absolute top-1 w-5 h-5 rounded-full bg-gradient-to-r from-primary to-purple-500"
                animate={{ x: billingPeriod === 'monthly' ? 4 : 25 }}
                transition={{ duration: 0.2 }}
              />
            </button>
            <span className={`text-sm font-medium ${billingPeriod === 'yearly' ? 'text-text-primary' : 'text-text-muted'}`}>
              Yearly
              <span className="ml-2 px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full">
                Save 20%
              </span>
            </span>
          </div>
        </motion.div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {plans.map((plan, index) => (
            <motion.div
              key={plan.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className={`relative ${plan.popular ? 'lg:-mt-4' : ''}`}
            >
              {/* Popular badge */}
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 -translate-x-1/2 z-10">
                  <span className="px-4 py-1.5 bg-gradient-to-r from-primary to-purple-500 text-white text-sm font-medium rounded-full flex items-center gap-2 shadow-lg shadow-primary/30">
                    <Crown className="w-4 h-4" />
                    Most Popular
                  </span>
                </div>
              )}

              {/* Card */}
              <Card
                variant={plan.popular ? 'default' : 'bordered'}
                className={`h-full flex flex-col ${
                  plan.popular
                    ? 'border-2 border-primary shadow-xl shadow-primary/20 relative overflow-hidden'
                    : ''
                } ${plan.popular ? 'p-8' : 'p-6'}`}
              >
                {/* Popular glow */}
                {plan.popular && (
                  <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent" />
                )}

                <div className="relative z-10">
                  {/* Plan header */}
                  <div className="flex items-center gap-4 mb-6">
                    <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${plan.gradient} flex items-center justify-center ${plan.popular ? 'shadow-lg' : ''}`}>
                      {plan.name === 'Enterprise' ? (
                        <Crown className="w-7 h-7 text-white" />
                      ) : plan.name === 'Pro' ? (
                        <Zap className="w-7 h-7 text-white" />
                      ) : (
                        <Shield className="w-7 h-7 text-white" />
                      )}
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-text-primary">{plan.name}</h3>
                      <p className="text-text-muted text-sm">{plan.limit}</p>
                    </div>
                  </div>

                  <p className="text-text-secondary text-sm mb-6">{plan.description}</p>

                  {/* Price */}
                  <div className="mb-6">
                    {plan.price === 0 ? (
                      <div className="flex items-baseline">
                        <span className="text-4xl font-bold text-text-primary">Free</span>
                      </div>
                    ) : (
                      <div className="flex items-baseline">
                        <span className="text-5xl font-bold text-text-primary">
                          ${billingPeriod === 'yearly' ? Math.round(plan.price * 0.8) : plan.price}
                        </span>
                        <span className="text-text-muted ml-2">/month</span>
                      </div>
                    )}
                    {billingPeriod === 'yearly' && plan.price > 0 && (
                      <p className="text-text-muted text-sm mt-1">
                        Billed annually (${Math.round(plan.price * 0.8 * 12)}/year)
                      </p>
                    )}
                  </div>

                  {/* Features list */}
                  <ul className="space-y-3 mb-8 flex-grow">
                    {plan.features.map((feature) => (
                      <li key={feature} className="flex items-start gap-3">
                        <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                        <span className="text-text-secondary text-sm">{feature}</span>
                      </li>
                    ))}
                  </ul>

                  {/* CTA Button */}
                  <Link href={plan.price === 0 || plan.name === 'Pro' ? "/signup" : "#contact"}>
                    <Button
                      variant={plan.popular ? 'primary' : 'secondary'}
                      className={`w-full ${plan.popular ? 'py-4' : ''}`}
                    >
                      {plan.cta}
                      {plan.popular && <ArrowRight className="w-4 h-4 ml-2" />}
                    </Button>
                  </Link>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Money Back Guarantee */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-16 text-center"
        >
          <div className="glass rounded-2xl p-8 max-w-2xl mx-auto border border-white/10">
            <div className="flex items-center justify-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                <Shield className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-lg font-semibold text-text-primary">30-Day Money-Back Guarantee</h3>
            </div>
            <p className="text-text-secondary">
              Try KERNELIZE risk-free. If you're not completely satisfied with our platform within the first 30 days, we'll refund your payment, no questions asked.
            </p>
          </div>
        </motion.div>

        {/* FAQ Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mt-20 max-w-3xl mx-auto"
        >
          <h3 className="text-2xl font-bold text-text-primary text-center mb-8">
            Frequently Asked Questions
          </h3>
          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <div
                key={index}
                className="glass rounded-xl overflow-hidden border border-white/5"
              >
                <button
                  onClick={() => setOpenFaq(openFaq === index ? null : index)}
                  className="w-full px-6 py-4 flex items-center justify-between text-left"
                >
                  <span className="font-medium text-text-primary">{faq.question}</span>
                  <motion.span
                    animate={{ rotate: openFaq === index ? 180 : 0 }}
                    className="text-text-muted"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </motion.span>
                </button>
                <motion.div
                  initial={false}
                  animate={{ height: openFaq === index ? 'auto' : 0, opacity: openFaq === index ? 1 : 0 }}
                  className="overflow-hidden"
                >
                  <div className="px-6 pb-4 text-text-secondary">
                    {faq.answer}
                  </div>
                </motion.div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
