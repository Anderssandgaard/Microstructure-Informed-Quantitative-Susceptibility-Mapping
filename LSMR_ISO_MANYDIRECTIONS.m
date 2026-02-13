function v = LSMR_ISO_MANYDIRECTIONS(u,params,dim)
    
if params.switch==1
  dim4   = size(params.Demag,4);
if dim == 1
  v      = zeros(size(params.Demag),'like',params.Demag);
  u      = reshape(u,[size(params.Demag,1),size(params.Demag,2),size(params.Demag,3)]);
  for i = 1:dim4
  temp1   = params.Mask1.*(real(ifftn(fftn(params.Mask2.*u).*params.Demag(:,:,:,i))));
  temp2   = params.Mask1.*(params.N_meso(:,:,:,i).*params.Mask2.*u);
 
  temp1   = params.Mask1.*(temp1 - 1*mean(temp1(params.Mask3==1)));
  temp2   = params.Mask1.*(temp2 - 1*mean(temp2(params.Mask3==1)));

  v(:,:,:,i) = temp1 + temp2;
  
  end
 % v = v;
  v = v(:);
elseif dim == 2
  v = zeros(size(params.Mask2),'like',params.Demag);
  u = reshape(u,size(params.Demag));
  for i=1:dim4

  temp = u(:,:,:,i);
  temp = params.Mask1.*(temp - mean(temp(params.Mask3==1)));

  temp1 =  params.Mask2.*real(ifftn(fftn(temp).*params.Demag(:,:,:,i)));     
  temp2 =  params.Mask2.*params.N_meso(:,:,:,i).*temp;     
  
  v = v + temp1 + temp2;
  end
  v = v(:); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif params.switch==0
  dim4   = size(params.Demag,4);  
if dim == 1
  v      = zeros(size(params.Demag),'like',params.Demag);
  u      = reshape(u,[size(params.Demag,1),size(params.Demag,2),size(params.Demag,3)]);
  for  i = 1:dim4
  temp1   = params.Mask1.*(real(ifftn(fftn(params.Mask2.*u).*params.Demag(:,:,:,i))));
 
  temp1   = params.Mask1.*(temp1 - 1*mean(temp1(params.Mask3==1)));
  
  v(:,:,:,i) = temp1;
  end
  v      = v(:);
elseif dim == 2
  v      = zeros(size(params.Mask2),'like',params.Demag);
  u      = reshape(u,size(params.Demag));
  for i=1:dim4

  temp = u(:,:,:,i);
  temp = params.Mask1.*(temp - mean(temp(params.Mask3==1)));

  temp1 =  params.Mask2.*real(ifftn(fftn(temp).*params.Demag(:,:,:,i)));     
  
  v = v + temp1;

  end
  v    = v(:); 
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




end