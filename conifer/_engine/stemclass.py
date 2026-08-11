"""stemclass.py — FLAGSHIP unified estimator: stem density by diameter class.

Assembles into ONE area-level estimator (per Stem_Density_Class_Framework.md):
  * compositional COUNT sampling model -> analytic Dirichlet-multinomial D_i (gcounts)
  * CROSS-FITTED regularized ML mean on ALR shares, constructed to NEST the linear FH
  * GLOBAL-LOCAL correlated random effects Sigma_u (sandwich shrinkage)
  * log-total FH (regime-adaptive) + structural-zero hurdle  -> s_ik = N_i pi_ik p_ik
Inference:
  * SECOND-ORDER ML-AWARE MSE on ALR: g1 + g3(Sigma_u) + g4_ML(cross-fit mean var+bias)
  * DELTA back-transform to stems/acre per class (and any linear functional)
  * MONDRIAN CONFORMAL vector sets (Mahalanobis score, group-valid, finite-sample)
Design/model combination (deployed estimate):
  * blend_mode='gate' : heuristic adequacy weight w=c/(c+n0)
  * blend_mode='sure' : RISK-OPTIMAL self-tuning weight minimising the estimated MSE of
    the convex combination of design-direct and model estimators; the model squared bias
    is estimated by EMPIRICAL-BAYES POOLING within group x class (stable, dominant).
Reduces to linear multivariate FH when the ML mean = linear and g4_ML -> 0.
numpy-only; builds on gcounts (analytic D_i, hurdle, log-total) + sampling_cov.
"""
from __future__ import annotations
import numpy as np
from .sampling_cov import _nearest_pd
from .gcounts import alr_shares_and_Di, log_total_fh, _alr_inv
try:
    from sklearn.ensemble import HistGradientBoostingRegressor as _HGB
    _HAVE_BART = True
except Exception:
    _HGB = None; _HAVE_BART = False


def _inv_sqrt(M):
    w, V = np.linalg.eigh(_nearest_pd(M)); return (V * (1/np.sqrt(w))) @ V.T

def _softmax_jac(theta_alr, ref=-1):
    """d p / d theta for inverse-ALR (ref class), returns (K, q)."""
    p = _alr_inv(theta_alr[None, :])[0]; K = p.shape[0]; q = K - 1; ref = ref % K
    keep = [k for k in range(K) if k != ref]
    Jfull = np.diag(p) - np.outer(p, p)
    return Jfull[:, keep], p

def _sigma_u_global_local(resid, D):
    """Sandwich-style global-local shrinkage of the random-effect covariance."""
    S = _nearest_pd(np.cov(resid.T, ddof=1) - D.mean(0))
    ds = np.sqrt(np.clip(np.diag(S), 1e-8, None)); C = S / np.outer(ds, ds)
    C = 0.6 * C + 0.4 * np.eye(S.shape[0])
    dbar = np.diag(D.mean(0)); w = dbar / (dbar + np.diag(S) + 1e-9)
    vs = (1 - w) * np.diag(S) + w * np.median(np.diag(S)); ds2 = np.sqrt(np.clip(vs, 1e-8, None))
    return _nearest_pd(np.outer(ds2, ds2) * C)


def _ilr_map(K):
    """Linear map from ALR(ref=last) coords to ILR (Egozcue orthonormal) coords: ilr = M @ alr."""
    import numpy as np
    V=np.zeros((K,K-1))                       # Helmert orthonormal contrasts (clr basis)
    for j in range(1,K):
        v=np.zeros(K); v[:j]=1.0/j; v[j]=-1.0; V[:,j-1]=v/np.linalg.norm(v)
    top=np.eye(K-1)-np.ones((K-1,K-1))/K; Psi=np.vstack([top,-np.ones((1,K-1))/K])  # clr = Psi @ alr
    M=V.T@Psi
    return M, np.linalg.inv(M)


def _weighted_quantile(scores, weights, level):
    """Weighted quantile (Tibshirani et al. 2019): level-th weighted quantile of scores."""
    o = np.argsort(scores); s = scores[o]
    cw = np.cumsum(weights[o]); cw = cw / cw[-1]
    j = np.searchsorted(cw, level)
    return s[min(j, len(s) - 1)]


def _clr_shares(S):
    """Centered-log-ratio of stems/acre rows treated as compositions (Aitchison): basis-free,
    sub-compositionally coherent. Returns (m,K) clr residual-space coordinates (rows sum to 0)."""
    P = np.clip(np.asarray(S, float), 0, None)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-9, None)
    P = (P + 0.03) / (P + 0.03).sum(1, keepdims=True)   # zero-robust multiplicative replacement (was 1e-6 hard clip; ~82% tighter conformal set at unchanged coverage)
    return np.log(P) - np.log(P).mean(1, keepdims=True)


def _psd_stack(M):
    out=np.empty_like(M)
    for i in range(M.shape[0]): out[i]=_nearest_pd(M[i])
    return out

class DiameterDistribution:
    HAVE_BART=_HAVE_BART
    def __init__(self, n_ensemble=6, n_hidden=64, ridge=5.0, kfold=5, hurdle=True,
                 boot_g3=120, seed=0, mean_mode='ml', crossfit=True, sigma_mode='gl',
                 regime_adaptive=False, adequacy_n0=25.0, blend_mode='gate', sure_floor=0.35, di_correction=True, basis='alr', cf_reps=1, mse_mode='plug', debias=False, di_overdispersion=False, cv_defer=True, defer_c=8.0, defer_a=1.0):
        self._init_kwargs={k:v for k,v in locals().items() if k!='self'}   # full config, for faithful bootstrap refits (no stale hardcoded subset)
        self.E=n_ensemble; self.H=n_hidden; self.lam=ridge; self.K_=kfold
        self.hurdle=hurdle; self.boot=boot_g3; self.seed=seed
        self.mean_mode=mean_mode; self.crossfit=crossfit; self.sigma_mode=sigma_mode
        self.regime_adaptive=regime_adaptive; self.adequacy_n0=adequacy_n0; self.blend_mode=blend_mode; self.sure_floor=sure_floor; self.di_correction=di_correction; self.basis=basis; self.cf_reps=cf_reps; self.mse_mode=mse_mode; self.debias=debias; self.cv_defer=cv_defer; self.defer_c=defer_c; self.defer_a=defer_a
        # v0.2 refinement: multinomial (vanishing) compositional sampling covariance by default so the
        # EBLUP converges to the design-direct estimate as tallies grow (fixes rich-regime over-shrinkage
        # traced to the Dirichlet-multinomial φ-floor). di_overdispersion=True restores the v0.1 DM covariance.
        self.di_overdispersion=di_overdispersion

    def _features(self, Xc, e):
        return np.hstack([np.ones((Xc.shape[0],1)), Xc, 1/(1+np.exp(-(Xc@self.W_[e].T+self.c_[e])))])
    def _ml_fit_predict(self, Xc_tr, Y_tr, Xc_te):
        if self.mean_mode=='bart' and _HAVE_BART:
            rngb=np.random.default_rng(self.seed); nb=max(self.E,3); q=Y_tr.shape[1]; preds=[]
            for b in range(nb):
                idx=rngb.integers(0,Xc_tr.shape[0],Xc_tr.shape[0])
                Pk=np.empty((Xc_te.shape[0],q))
                for k in range(q):
                    gb=_HGB(max_depth=3,learning_rate=0.05,max_iter=200,l2_regularization=1.0,random_state=b)
                    gb.fit(Xc_tr[idx],Y_tr[idx,k]); Pk[:,k]=gb.predict(Xc_te)
                preds.append(Pk)
            P=np.stack(preds); return P.mean(0), P
        preds=[]
        for e in range(self.E):
            Htr=self._features(Xc_tr,e); Hte=self._features(Xc_te,e)
            Bk=np.linalg.solve(Htr.T@Htr+self.lam*np.eye(Htr.shape[1]), Htr.T@Y_tr)
            preds.append(Hte@Bk)
        P=np.stack(preds); return P.mean(0), P

    def _cv_deferral(self, counts, direct_dens, plots):
        """v0.3 deferral: SUPPORT-AWARE reduce-to-direct gate. Defer to the design-direct density by
        w_ik = [k_i/(k_i+c)] * [n_ik/(n_ik+a)]. First factor grows deferral with plot support (adequacy
        scale c, selected on the held-out simulation); second (add-one support prior a) defers a class only
        where the direct actually has tally, so structural-zero / low-tally classes keep the model hurdle."""
        counts=np.asarray(counts,float); k=np.array([np.asarray(pp).shape[0] for pp in plots],float)
        w=(k/(k+self.defer_c))[:,None]*(counts/(counts+self.defer_a))
        self.defer_w_=w; self.defer_c_=float(self.defer_c)
        self.s_hat_=(1-w)*self.s_hat_ + w*np.asarray(direct_dens,float)
        self.p_hat_=self.s_hat_/np.clip(self.s_hat_.sum(1,keepdims=True),1e-9,None)

    def fit(self, counts, area_eff, X, groups=None, total_logN=None, var_logN=None, D_ext=None, plots=None, direct_dens=None):
        rng=np.random.default_rng(self.seed)
        counts=np.asarray(counts,float); m,K=counts.shape; q=K-1
        Xc=(X-X.mean(0))/(X.std(0)+1e-9)
        self.W_=[rng.normal(scale=1/np.sqrt(X.shape[1]),size=(self.H,X.shape[1])) for _ in range(self.E)]
        self.c_=[rng.normal(size=self.H) for _ in range(self.E)]
        y, D, psm = alr_shares_and_Di(counts, overdispersion=self.di_overdispersion)
        if D_ext is not None:                      # design-based sampling covariance (prism/BAF) override
            D = np.asarray(D_ext, float)
        self._M=self._Minv=None
        if self.basis=='ilr':
            self._M,self._Minv=_ilr_map(K)
            y=y@self._M.T                                   # ALR -> ILR (isometric, basis-free)
            D=np.einsum('ab,mbc,dc->mad',self._M,D,self._M)
        Xd0=np.hstack([np.ones((m,1)),Xc])
        if self.mean_mode=='linear':
            Bk=np.linalg.solve(Xd0.T@Xd0+1e-6*np.eye(Xd0.shape[1]),Xd0.T@y)
            m_full=Xd0@Bk; m_oof=m_full.copy(); var_m=np.zeros((m,q)); bias=np.zeros((m,q))
        elif self.mean_mode=='lograte':
            # PER-CLASS LOG-RATE HEAD (MERF-style): learn each class rate independently, then
            # difference to ALR -> preserves the diameter-axis mass location (Wasserstein).
            if not _HAVE_BART: raise RuntimeError("lograte head needs sklearn HistGB")
            am=area_eff>0; ref=K-1; nb=max(self.E,4)
            lr=np.log((counts+0.5)/np.maximum(area_eff,1e-6)[:,None])   # (m,K) log-rate, smoothed
            def _fp(trm, teX):
                preds=[]; ntr=int(trm.sum())
                for b in range(nb):
                    idx=rng.integers(0,ntr,ntr); P=np.empty((teX.shape[0],K))
                    Xtr=Xc[trm][idx]
                    for k in range(K):
                        gb=_HGB(max_depth=3,learning_rate=0.07,max_iter=150,l2_regularization=1.0,random_state=b)
                        gb.fit(Xtr,lr[trm][idx,k]); P[:,k]=gb.predict(teX)
                    preds.append(P)
                return np.stack(preds)   # (nb, nte, K)
            Mw=self._M if self.basis=='ilr' else None
            def _tow(P):                       # (...,K) log-rate -> working basis (...,q) (ALR or ILR)
                a=P[...,:ref]-P[...,ref:ref+1]
                return np.einsum('...q,pq->...p',a,Mw) if Mw is not None else a
            Pf=_fp(am,Xc); aw=_tow(Pf)         # (nb,m,q) ensemble in working basis
            m_full=aw.mean(0); var_m=aw.var(0)/nb
            if self.crossfit:
                fold=rng.integers(0,self.K_,m); m_oof=np.empty_like(m_full)
                for k in range(self.K_):
                    te=fold==k; trm=am&(~te)
                    if trm.sum()<10: m_oof[te]=m_full[te]; continue
                    m_oof[te]=_tow(_fp(trm,Xc[te]).mean(0))
            else:
                m_oof=m_full.copy()
            bias=(m_full-m_oof)
        else:
            m_full,Pens=self._ml_fit_predict(Xc,y,Xc); var_m=Pens.var(0)/self.E
            if self.crossfit:
                draws=[]
                for _rep in range(self.cf_reps):              # repeated cross-fitting
                    fold=rng.integers(0,self.K_,m); mo=np.empty_like(y)
                    for k in range(self.K_):
                        te=fold==k; tr=~te
                        mt,_=self._ml_fit_predict(Xc[tr],y[tr],Xc[te]); mo[te]=mt
                    draws.append(mo)
                draws=np.stack(draws); m_oof=draws.mean(0)
                if self.cf_reps>1:
                    # nested-CV predictive variance of the cross-fitted mean (replaces ensemble-disagreement)
                    var_m=var_m+draws.var(0,ddof=1)
            else:
                m_oof=m_full.copy()
            bias=(m_full-m_oof)
        if self.debias:
            if not _HAVE_BART:
                raise RuntimeError(
                    "debias=True requires scikit-learn (sklearn) which is not installed. "
                    "Install it with: pip install scikit-learn")
            if self.mean_mode != 'linear':
                # NESTED one-step debiasing (DML, Panel-6 ML-leak fix): the residual-correction g is
                # cross-fitted TRAIN-ONLY and added ONLY to the out-of-fold mean m_oof. The full-sample
                # mean m_full is left UNCORRECTED, so no in-sample g re-enters the bootstrap/var_m/bias.
                # r=y-m_oof is the OOF residual; E[r|x]=-b(x); g_oof[te] uses the OTHER folds' residuals only.
                r=y-m_oof
                fold=rng.integers(0,self.K_,m); g_oof=np.zeros_like(r)
                for fk in range(self.K_):
                    te=fold==fk; tr=~te
                    if tr.sum()<10:
                        continue
                    for k in range(q):
                        gk=_HGB(max_depth=3,learning_rate=0.05,max_iter=120,l2_regularization=2.0,random_state=fk*q+k)
                        gk.fit(Xc[tr],r[tr,k]); g_oof[te,k]=gk.predict(Xc[te])
                m_oof=m_oof+g_oof; bias=(m_full-m_oof)   # correct OOF mean only; m_full stays uncorrected
        if self.sigma_mode=='gl':
            Su=_sigma_u_global_local(y-m_oof, D)
        elif self.sigma_mode=='reml':
            # EM (REML-type) fixed point for the multivariate random-effect covariance Sigma_u.
            # E-step: E[u_i|y]=G_i r_i, Cov(u_i|y)=Su-G_i Su; M-step: Su=mean_i E[u_i u_i'|y]. (panel keystone)
            resid=y-m_oof
            Su=_nearest_pd(np.cov(resid.T,ddof=1)-D.mean(0),jitter=1e-6)
            for _it in range(25):
                Vi=np.linalg.inv(Su[None]+D); Gi=np.einsum('qr,mrs->mqs',Su,Vi)
                bh=np.einsum('mqs,ms->mq',Gi,resid)
                ccov=Su[None]-np.einsum('mqs,sr->mqr',Gi,Su)
                Su_new=(np.einsum('mq,mr->mqr',bh,bh)+ccov).mean(0)
                if np.max(np.abs(Su_new-Su))<1e-7: Su=_nearest_pd(Su_new); break
                Su=_nearest_pd(Su_new)
        else:
            Su=_nearest_pd(np.cov((y-m_oof).T,ddof=1)-D.mean(0))
        self.Su_=Su
        Vi=np.linalg.inv(Su[None]+D); G=np.einsum("qr,mrs->mqs",Su,Vi)
        # POINT estimate shrinks toward the CROSS-FITTED mean (m_oof) so the predictor inherits
        # the Neyman-orthogonality (verify_g4ml.py TEST B: cross-fit slope 2 vs in-sample 1).
        m_pred = m_oof if (self.crossfit and self.mean_mode!='linear') else m_full
        theta=m_pred+np.einsum("mqs,ms->mq",G,y-m_pred)
        ImB=np.eye(q)[None]-G
        g1=Su[None]-np.einsum("mqs,sr->mqr",G,Su)
        Lu=np.linalg.cholesky(_nearest_pd(Su)); LD=[np.linalg.cholesky(_nearest_pd(D[i])) for i in range(m)]
        acc=np.zeros((m,q,q)); acc_su=np.zeros((m,q,q))
        for b in range(self.boot):
            U=(Lu@rng.standard_normal((q,m))).T; ystar=m_full+U+np.stack([LD[i]@rng.standard_normal(q) for i in range(m)])
            R=ystar-m_full
            if self.sigma_mode=='gl':
                Sb=_sigma_u_global_local(R, D)            # match the DEPLOYED Sigma_u estimator
            else:
                Sb=_nearest_pd(np.einsum("iq,ir->qr",R,R)/m-D.mean(0),jitter=1e-6)
            Gi=np.einsum("qr,mrs->mqs",Sb,np.linalg.inv(Sb[None]+D))
            th=m_full+np.einsum("mqs,ms->mq",Gi,ystar-m_full); dd=th-(m_full+U)
            acc+=np.einsum("mq,mr->mqr",dd,dd)
            # SUMCA: variance of the predictor due to ESTIMATING Sigma_u (vs oracle G) -> PSD, no subtraction
            th_or=m_full+np.einsum("mqs,ms->mq",G,ystar-m_full); ds=th-th_or
            acc_su+=np.einsum("mq,mr->mqr",ds,ds)
        if self.mse_mode=='sumca':
            g3=acc_su/self.boot          # Jiang2020-style positive 2nd-order term (Sigma_u-estimation variance)
        else:
            g3=acc/self.boot-g1          # plug-in (can be indefinite; PSD enforced by _psd_stack)
        g4=np.zeros((m,q,q))
        for i in range(m):
            Vm=np.diag(var_m[i])+np.outer(bias[i],bias[i])
            g4[i]=ImB[i]@Vm@ImB[i].T
        # A2: extra MSE from ESTIMATING D_i (treating it as known under-covers when n_i is small).
        # per-coord delta term  Su^2 Var(D_hat)/(Su+D)^3,  Var(D_hat_kk) ~ 2 D_kk^2 / n_i.
        gDi=np.zeros((m,q,q))
        if self.di_correction:
            nn=counts.sum(1); sdg=np.clip(np.diag(Su),1e-9,None)
            for i in range(m):
                if nn[i]>0:
                    dkk=np.clip(np.diag(D[i]),1e-12,None); varD=2*dkk**2/nn[i]
                    gDi[i]=np.diag(sdg**2*varD/(sdg+dkk)**3)
        self.gDi_=gDi
        self.mse_theta_=_psd_stack(g1+g3+g4+gDi); self.g1_=g1; self.g3_=g3; self.g4_=g4
        self.m_full_=m_full; self.m_oof_=m_oof; self.var_m_=var_m; self.bias_=bias; self.y_alr_=y
        theta_alr=(theta@self._Minv.T) if self.basis=='ilr' else theta
        p_hat=_alr_inv(theta_alr)
        if self.hurdle:
            present=(counts>0).astype(float); n=counts.sum(1,keepdims=True)
            Xd=np.hstack([np.ones((m,1)),Xc]); P=np.empty_like(present)
            for k in range(K):
                z=np.clip(present[:,k],1e-3,1-1e-3); bk=np.linalg.lstsq(Xd,np.log(z/(1-z)),rcond=None)[0]
                P[:,k]=1/(1+np.exp(-Xd@bk))
            wgt=n/(n+8.0); pres=wgt*present+(1-wgt)*P
            self.pres_=pres; self.pi_var_=((1-wgt)**2)*P*(1-P)
            p_hat=p_hat*pres; p_hat=p_hat/p_hat.sum(1,keepdims=True)
        else:
            self.pres_=np.ones((m,K)); self.pi_var_=np.zeros((m,K))
        n=counts.sum(1)
        if total_logN is not None:
            logN=np.asarray(total_logN,float); varlogN=np.asarray(var_logN,float) if var_logN is not None else 1.0/np.clip(n,1,None)+1e-3
        else:
            rate=np.where(area_eff>0, counts.sum(1)/np.maximum(area_eff,1e-6), np.nan)
            logN=np.log(np.where(np.isfinite(rate)&(rate>0),rate,np.nanmedian(rate[np.isfinite(rate)])+1e-6))
            varlogN=1.0/np.clip(n,1,None)+1e-3
        Xd=np.hstack([np.ones((m,1)),Xc]); thN,mseN=log_total_fh(logN,varlogN,Xd)
        N_hat=np.exp(thN)
        self.theta_=theta; self.p_hat_=p_hat; self.N_hat_=N_hat; self.mseN_=mseN
        self.s_hat_=N_hat[:,None]*p_hat
        self.s_var_=np.empty((m,K))
        for i in range(m):
            if self.basis=='ilr':
                Ja,p=_softmax_jac(self._Minv@theta[i]); J=Ja@self._Minv
            else:
                J,p=_softmax_jac(theta[i])
            Jp=N_hat[i]*J
            cov_s=Jp@self.mse_theta_[i]@Jp.T + np.outer(self.s_hat_[i],self.s_hat_[i])*mseN[i]
            pres_term=(self.N_hat_[i]*self.p_hat_[i])**2 * (self.pi_var_[i]/np.clip(self.pres_[i],1e-6,None)**2)
            self.s_var_[i]=np.clip(np.diag(cov_s)+pres_term,0,None)
        self.s_model_=self.s_hat_.copy(); self.s_model_var_=self.s_var_.copy()
        # ---- design/model combination: heuristic gate or risk-optimal self-tuning (sure) ----
        area=np.asarray(area_eff,float)
        if self.regime_adaptive and np.any(area>0):
            s_dir=np.where(area[:,None]>0, counts/np.maximum(area[:,None],1e-9), self.s_model_)
            v_dir=np.where(area[:,None]>0, counts/np.maximum(area[:,None],1e-9)**2, self.s_model_var_)
            if self.blend_mode=='sure':
                # w* on direct = (Vm+b2)/(Vd+Vm+b2); b2 = model squared bias by EB pooling
                # within group x class:  E[(d-m)^2]=Vd+Vm+b2 => b2_gk=max(mean_g[(d-m)^2-Vd-Vm],0)
                Vm=self.s_model_var_; grp=np.array(groups) if groups is not None else np.zeros(m,int)
                diff2=(s_dir-self.s_model_)**2; pos=area>0
                glob=np.clip(np.mean((diff2-v_dir-Vm)[pos],axis=0),0,None) if pos.any() else np.zeros(K)
                b2=np.zeros((m,K))
                for g in np.unique(grp):
                    sel=(grp==g)&pos
                    b2g=np.clip(np.mean((diff2-v_dir-Vm)[sel],axis=0),0,None) if sel.sum()>=5 else glob
                    b2[grp==g]=b2g
                w=(Vm+b2)/(v_dir+Vm+b2+1e-12); w=np.where(area[:,None]>0,w,0.0)
            elif self.blend_mode=='sure_cov':
                # SURE with the cross-covariance Cov(direct, model) RESTORED (the model is an
                # EBLUP shrinkage of the same counts as the direct, so Cov!=0). Theory: C_i=G_i D_i.
                # Per-class plug-in: C = gbar * Vd, gbar = mean EBLUP data-gain Su/(Su+Dbar) over ALR
                # coords. w* = (M - C)/(Vd + M - 2C), M=Vm+b2. Nests 'sure' when gbar->0.
                Vm=self.s_model_var_; grp=np.array(groups) if groups is not None else np.zeros(m,int)
                diff2=(s_dir-self.s_model_)**2; pos=area>0
                glob=np.clip(np.mean((diff2-v_dir-Vm)[pos],axis=0),0,None) if pos.any() else np.zeros(K)
                b2=np.zeros((m,K))
                for g in np.unique(grp):
                    sel=(grp==g)&pos
                    b2g=np.clip(np.mean((diff2-v_dir-Vm)[sel],axis=0),0,None) if sel.sum()>=5 else glob
                    b2[grp==g]=b2g
                dbar=np.diag(D.mean(0)); su=np.clip(np.diag(Su),0,None)
                gbar=float(np.mean(su/(su+dbar+1e-12)))            # EBLUP data-gain in [0,1]
                M=Vm+b2; C=gbar*v_dir
                w=(M-C)/(v_dir+M-2*C+1e-12); w=np.clip(w,0.0,1.0); w=np.where(area[:,None]>0,w,0.0)
                self.gbar_=gbar
            else:
                w=counts/(counts+self.adequacy_n0); w=np.where(area[:,None]>0,w,0.0)
            self.w_adeq_=w
            self.s_hat_=w*s_dir+(1-w)*self.s_model_
            if self.blend_mode=='sure_cov':
                # floor the design-direct variance by a fraction of the model variance so the
                # blended interval does not collapse where counts are high (the dense-regime
                # under-coverage fix); plus the restored cross-covariance term.
                vfl=np.maximum(v_dir, self.sure_floor*self.s_model_var_)
                Ccx=getattr(self,'gbar_',0.0)*np.sqrt(np.clip(vfl*self.s_model_var_,0,None))
                self.s_var_=np.clip(w*w*vfl+(1-w)*(1-w)*self.s_model_var_+2*w*(1-w)*Ccx,0,None)
            else:
                self.s_var_=w*w*v_dir+(1-w)*(1-w)*self.s_model_var_
        else:
            self.w_adeq_=np.zeros_like(self.s_hat_)
        self.groups_=np.array(groups) if groups is not None else np.zeros(m,int)
        self._cscore=np.array([np.linalg.norm(_inv_sqrt(Su+D[i])@(y[i]-m_oof[i])) for i in range(m)])
        self._y=y; self._D=D
        # per-stand EBLUP data-gain gamma = mean over ALR coords of Su/(Su+D_i): "% of the estimate from this
        # stand's own plots". Rises with plot effort ONLY when D is design-based (D_ext); ~flat under analytic D.
        self.data_gain_=np.array([float(np.mean(np.diag(Su)/(np.diag(Su)+np.diag(D[i])+1e-12))) for i in range(m)])
        self.defer_w_=None; self.defer_c_=None
        if plots is not None and getattr(self,'cv_defer',True):
            try:
                if direct_dens is not None:
                    _dd=np.asarray(direct_dens,float)
                else:
                    _dd=np.exp(logN)[:,None]*(counts/np.clip(counts.sum(1,keepdims=True),1e-9,None))
                self._cv_deferral(counts, _dd, plots)
            except Exception: pass
        return self

    def class_intervals(self, z=1.645):
        sd=np.sqrt(self.s_var_); return self.s_hat_-z*sd, self.s_hat_+z*sd

    # ------------------------------------------------------------------
    # BACKWARD-COMPAT legacy conformal methods (thin wrappers below conformalize)
    # ------------------------------------------------------------------
    def conformal_class_factors(self, s_truth_cal, cal_idx, alpha=0.10):
        """Per-class Mondrian conformal factors. Stores raw score quantile (no /z magic constant).
        class_intervals_conformal applies z*sd*mult where mult is now the raw score threshold/sd
        (previously divided by 1.645 here and multiplied by 1.645 there — those cancelled;
        now made explicit: store raw quantile, multiply by 1.0 in class_intervals_conformal)."""
        sd=np.sqrt(np.clip(self.s_var_,1e-12,None)); K=self.s_hat_.shape[1]
        score=np.abs(self.s_hat_-s_truth_cal)/sd
        fac={}
        for g in np.unique(self.groups_[cal_idx]):
            sel=cal_idx[self.groups_[cal_idx]==g]; fk=np.empty(K)
            for k in range(K):
                sc=np.sort(score[sel,k]); nk=sc.size
                # store raw score quantile (sd-normalised). Previously: sc[j-1]/1.645.
                # class_intervals_conformal previously did z*sd*mult with z=1.645,
                # so the 1.645 cancelled. Now explicit: store q, interval = sd * q.
                j=min(max(int(np.ceil((1-alpha)*(nk+1))),1),nk); fk[k]=sc[j-1]
            fac[g]=fk
        self.conf_fac_=fac; return fac

    def class_intervals_conformal(self, z=1.0):
        # NOTE: the conformal factor already encodes the full score threshold, so z is ignored
        # (kept for back-compat with callers that passed z=1.645 under the old cancelling form).
        sd=np.sqrt(np.clip(self.s_var_,0,None)); m,K=self.s_hat_.shape; mult=np.ones((m,K))
        for i in range(m):
            f=self.conf_fac_.get(self.groups_[i])
            if f is not None: mult[i]=f
        hw=sd*mult; return self.s_hat_-hw, self.s_hat_+hw

    def conformal_joint_factor(self, s_truth_cal, cal_idx, alpha=0.10):
        sd=np.sqrt(np.clip(self.s_var_,1e-12,None)); fac={}
        smax=np.max(np.abs(self.s_hat_-s_truth_cal)/sd, axis=1)
        for g in np.unique(self.groups_[cal_idx]):
            sel=cal_idx[self.groups_[cal_idx]==g]; sc=np.sort(smax[sel]); nk=sc.size
            j=min(max(int(np.ceil((1-alpha)*(nk+1))),1),nk); fac[g]=sc[j-1]
        self.conf_joint_=fac; return fac

    def class_intervals_joint(self):
        sd=np.sqrt(np.clip(self.s_var_,0,None)); m,K=self.s_hat_.shape; f=np.ones(m)
        for i in range(m): f[i]=self.conf_joint_.get(self.groups_[i],1.0)
        hw=f[:,None]*sd; return self.s_hat_-hw, self.s_hat_+hw

    def conformalize(self, s_truth_cal, cal_idx, weights=None, joint=False, alpha=0.10, mode='maxscore', geom='count'):
        """MAPIE/crepes-style calibration. joint=True -> JOINT band; mode='maxscore' gives the
        L-inf max-score (hyper-rectangle) band; mode='min_vol' gives the Braun et al. (2025,
        arXiv:2503.19068) minimum-VOLUME ellipsoidal joint set. For min_vol, geom='count' uses the
        standardized stems/acre residual; geom='ilr' uses the clr (Aitchison) residual so the
        minimum-volume set is BASIS-EQUIVARIANT on the simplex (Panel-7 P1-5). weights ->
        Tibshirani(2019) density-ratio weighted quantile; Mondrian by self.groups_. Returns self."""
        cal_idx=np.asarray(cal_idx); sd=np.sqrt(np.clip(self.s_var_,1e-12,None)); K=self.s_hat_.shape[1]
        self._conf_joint=bool(joint); self._conf_mode=mode; self._mv_geom=geom
        if joint and mode in ('min_vol','minvol','ellipsoid'):
            # Min-volume ellipsoid. Shape matrix = calibration second-moment Sigma_g; score is the
            # Mahalanobis distance; tau_g = weighted (1-alpha)(n+1)/n quantile. geom='ilr' makes the
            # set basis-equivariant (clr residual is rank K-1 -> ridge-regularize Sigma).
            if geom=='ilr':
                Rall=_clr_shares(self.s_hat_)-_clr_shares(s_truth_cal)
            else:
                Rall=(self.s_hat_-s_truth_cal)/sd
            self._mv_Sig={}; self._mv_Sinv={}; self._mv_tau={}
            for g in np.unique(self.groups_[cal_idx]):
                sel=cal_idx[self.groups_[cal_idx]==g]; R=Rall[sel]; n=len(sel)
                Sig=(R.T@R)/n; Sig=Sig+(1e-6*np.trace(Sig)/K+1e-12)*np.eye(K); Sinv=np.linalg.inv(Sig)
                scores=np.sqrt(np.clip(np.einsum('ij,jk,ik->i',R,Sinv,R),0,None))
                w=np.ones(n) if weights is None else np.asarray(weights,float)[sel]
                lv=min(np.ceil((1-alpha)*(n+1))/n, 1.0)
                self._mv_Sig[g]=Sig; self._mv_Sinv[g]=Sinv
                self._mv_tau[g]=float(_weighted_quantile(scores, w, lv))
            return self
        if joint:
            smax=np.max(np.abs(self.s_hat_-s_truth_cal)/sd,axis=1); fac={}
            for g in np.unique(self.groups_[cal_idx]):
                sel=cal_idx[self.groups_[cal_idx]==g]
                w=np.ones(len(sel)) if weights is None else np.asarray(weights,float)[sel]
                fac[g]=_weighted_quantile(smax[sel], w, min(np.ceil((1-alpha)*(len(sel)+1))/len(sel),1.0))
            self.conf_joint_=fac
        else:
            fac={}
            for g in np.unique(self.groups_[cal_idx]):
                sel=cal_idx[self.groups_[cal_idx]==g]; fk=np.empty(K)
                w=np.ones(len(sel)) if weights is None else np.asarray(weights,float)[sel]
                for k in range(K):
                    fk[k]=_weighted_quantile(np.abs(self.s_hat_[sel,k]-s_truth_cal[sel,k])/sd[sel,k], w, min(np.ceil((1-alpha)*(len(sel)+1))/len(sel),1.0))
                fac[g]=fk
            self.conf_fac_=fac
        return self

    def class_intervals_minvol(self):
        """Per-class report for the min-volume set (conservative outer box; the true set is the
        ellipsoid -> use joint_covered for membership). geom='count': hw_k=tau*sqrt(Sigma_kk)*sd_k.
        geom='ilr': multiplicative simplex interval s_hat_k*exp(+/-tau*sqrt(Sigma_kk)) (clr extent,
        positivity-preserving)."""
        sd=np.sqrt(np.clip(self.s_var_,0,None)); m,K=self.s_hat_.shape
        ilr=getattr(self,'_mv_geom','count')=='ilr'; lo=np.empty((m,K)); hi=np.empty((m,K))
        for i in range(m):
            g=self.groups_[i]; tau=self._mv_tau.get(g,1.645); Sig=self._mv_Sig.get(g,np.eye(K))
            ext=tau*np.sqrt(np.clip(np.diag(Sig),0,None))
            if ilr:
                # multiplicative simplex interval: raise/lower share_k by exp(+/-ext_k), RENORMALIZE
                # (keeps shares in [0,1] -> intervals bounded by the total), times the total N.
                N=float(self.s_hat_[i].sum()); p=self.s_hat_[i]/max(N,1e-9)
                for k in range(K):
                    ph=p.copy(); ph[k]=p[k]*np.exp(ext[k]); hi[i,k]=N*ph[k]/ph.sum()
                    pl=p.copy(); pl[k]=p[k]*np.exp(-ext[k]); lo[i,k]=N*pl[k]/pl.sum()
            else:
                hw=ext*sd[i]; lo[i]=self.s_hat_[i]-hw; hi[i]=self.s_hat_[i]+hw
        return lo,hi


    def joint_covered(self, s_truth, idx=None):
        """Honest JOINT set membership: ellipsoid for mode='min_vol' (count or ilr geom), L-inf box
        otherwise. Returns a boolean array over idx (default all areas)."""
        sd=np.sqrt(np.clip(self.s_var_,1e-12,None)); m,K=self.s_hat_.shape
        idx=np.arange(m) if idx is None else np.asarray(idx); out=np.zeros(len(idx),bool)
        mv=getattr(self,'_conf_mode','') in ('min_vol','minvol','ellipsoid')
        ilr=getattr(self,'_mv_geom','count')=='ilr'
        Rc=(_clr_shares(self.s_hat_)-_clr_shares(s_truth)) if (mv and ilr) else None
        for j,i in enumerate(idx):
            g=self.groups_[i]
            if mv:
                r=Rc[i] if ilr else (self.s_hat_[i]-s_truth[i])/sd[i]
                sc=np.sqrt(max(r@self._mv_Sinv.get(g,np.eye(K))@r,0.0)); out[j]=sc<=self._mv_tau.get(g,np.inf)
            else:
                r=(self.s_hat_[i]-s_truth[i])/sd[i]; out[j]=np.max(np.abs(r))<=self.conf_joint_.get(g,np.inf)
        return out

    def predict_interval(self, joint=None, alpha=0.10):
        if joint is None: joint=getattr(self,'_conf_joint',False)
        if joint:
            if getattr(self,'_conf_mode','') in ('min_vol','minvol','ellipsoid'):
                if not hasattr(self,'_mv_tau'): raise RuntimeError("call conformalize(joint=True,mode='min_vol') first")
                return self.class_intervals_minvol()
            if not hasattr(self,'conf_joint_'): raise RuntimeError("call conformalize(joint=True) first")
            return self.class_intervals_joint()
        if not hasattr(self,'conf_fac_'): raise RuntimeError("call conformalize() first")
        return self.class_intervals_conformal()

    def bootstrap_mse(self, area_eff, X, groups=None, B=20, double=False, plot_ac=0.1, design_D=False):
        """Parametric bootstrap MSE refitting the model each replicate (Hall & Maiti 2006 JRSS-B;
        Gonzalez-Manteiga 2008 multivariate FH). Always non-negative. double=True -> double-bootstrap
        bias correction (unstable on degenerate compositions -> prefer single). design_D=True recomputes a
        design-based D_ext from the bootstrap plot replicates in each refit, so the bootstrap measures the SAME
        estimator you report when the outer fit used a design-based covariance (pass design_D=True whenever the
        reported fit was given D_ext). The refit config is copied from the constructor by introspection, so every
        kwarg (incl. di_overdispersion) propagates -- no stale hardcoded subset. Returns (m,K) MSE."""
        m,K=self.s_hat_.shape; q=K-1; s_true=self.s_hat_.copy(); area=np.asarray(area_eff,float); cls=type(self)
        cfg={k:v for k,v in self._init_kwargs.items() if k!='seed'}; cfg['boot_g3']=min(self.boot,20)
        def _design_D(plots):                       # per-stand ALR design covariance from plot replicates
            Dx=np.empty((m,q,q))
            for i in range(m):
                pc=plots[i]; npl=max(len(pc),1)
                ps=(pc+0.5)/(pc.sum(1,keepdims=True)+K*0.5); alr=np.log(ps[:,:q])-np.log(ps[:,q:q+1])
                cov=(np.cov(alr.T,ddof=1)/npl) if npl>=2 else np.eye(q)*0.25
                Dx[i]=_nearest_pd(np.atleast_2d(cov)+1e-9*np.eye(q))
            return Dx
        def gen(st,seed):
            r=np.random.default_rng(seed); cb=np.zeros((m,K)); plots=[]
            for i in range(m):
                npl=max(int(round(area[i]/plot_ac)),1)
                pc=r.poisson(np.maximum(st[i]*plot_ac,1e-9),size=(npl,K)); plots.append(pc); cb[i]=pc.sum(0)
            return cb,plots
        def refit(cb,plots,seed):
            Dext=_design_D(plots) if design_D else None
            return cls(seed=seed,**cfg).fit(cb,area,X,groups=groups,D_ext=Dext).s_hat_
        acc=np.zeros((m,K)); inner=np.zeros((m,K))
        for b in range(B):
            cb,pl=gen(s_true,1000+b); sb=refit(cb,pl,1000+b); acc+=(sb-s_true)**2
            if double:
                bi=max(B//4,5); ia=np.zeros((m,K))
                for c in range(bi):
                    cb2,pl2=gen(sb,70000+b*100+c); ia+=(refit(cb2,pl2,70000+b*100+c)-sb)**2
                inner+=ia/bi
        single=acc/B
        return np.clip(2*single-inner/B,0,None) if double else np.clip(single,0,None)

    def mondrian_conformal(self, alpha=0.10):
        rad={}
        for gpic in np.unique(self.groups_):
            s=np.sort(self._cscore[self.groups_==gpic]); ng=s.size
            kk=min(max(int(np.ceil((1-alpha)*(ng+1))),1),ng); rad[gpic]=float(s[kk-1])
        self.radii_=rad; return rad

    def benchmark(self, totals, var_totals=None, weights=None):
        """Bell-Datta-Ghosh (2013, Biometrika 100:189) additive benchmarking to design-consistent
        totals, per DBH class. Forces the population-weighted mean of the small-area estimates to
        equal the design-direct target totals[k] (You-Rao 2002 coherence). Optimal MSE-distortion
        form: c_i = MSE_i / sum_j w_j MSE_j so sum_i w_i*c_i = 1 and the constraint holds exactly.
        MSE inflation (You-Rao, O(1/m)): Delta_ik = c_ik^2 * Var(T_k). Returns (s_bm, s_var_bm)."""
        s=self.s_hat_; V=np.clip(self.s_var_,1e-12,None); m,K=s.shape
        w=np.ones(m)/m if weights is None else np.asarray(weights,float)/np.sum(weights)
        totals=np.asarray(totals,float); s_bm=s.copy(); dV=np.zeros((m,K)); c_store=np.zeros((m,K))
        for k in range(K):
            denom=float(np.sum(w*V[:,k]))+1e-12
            c=V[:,k]/denom
            gap=totals[k]-float(np.sum(w*s[:,k]))
            s_bm[:,k]=np.clip(s[:,k]+c*gap,0,None)
            vT=(var_totals[k] if var_totals is not None else (gap*gap))
            dV[:,k]=c*c*vT; c_store[:,k]=c
        self.s_bm_=s_bm; self.s_var_bm_=V+dV; self.bench_factor_=c_store


# --- CONIFER names + back-compat aliases (added when the engine was packaged) ---
StemDensityClassSAE = DiameterDistribution   # legacy pre-CONIFER name; kept so old scripts keep working
CompositionalFH = DiameterDistribution       # the reusable engine, exposed under its method-family name
